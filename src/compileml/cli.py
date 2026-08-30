"""compileml — command-line interface.

Artifact-side commands (inspect / verify / score / export) use only the
standard library plus the runtime; validate and compile pull in the
compile-side dependencies on demand.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from compileml.runtime import decide, load_artifact
from compileml.runtime.io import ArtifactError


def _read_csv(path: str, feature_names: list[str], y_col: str | None = None):
    """Rows (list of float-or-None per feature) and optional labels from a CSV."""
    rows, labels = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [n for n in feature_names if n not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"CSV is missing feature columns: {missing}")
        if y_col and y_col not in (reader.fieldnames or []):
            raise SystemExit(f"CSV is missing outcome column {y_col!r}")
        for record in reader:
            row = []
            for name in feature_names:
                raw = (record.get(name) or "").strip()
                row.append(float(raw) if raw else None)
            rows.append(row)
            if y_col:
                labels.append(int(float(record[y_col])))
    return rows, labels


# ---------------------------------------------------------------- commands
def cmd_inspect(args) -> int:
    artifact = load_artifact(args.artifact, verify=not args.no_verify)
    meta = artifact.get("metadata", {})
    summary = {
        "artifact_hash": artifact.get("artifact_hash"),
        "schema_version": artifact.get("schema_version"),
        "n_features": artifact["model"]["n_features"],
        "n_trees": len(artifact["model"]["trees"]),
        "whitebox_max_depth": artifact["runtime"].get("whitebox_max_depth"),
        "exact_attribution": artifact["runtime"].get("exact_attribution"),
        "input_precision": artifact["model"].get("input_precision", "float64"),
        "scale": artifact["scale"],
        "micro_scale": artifact["model"]["micro_scale"],
        "bands": {
            "labels": artifact["bands"]["labels"],
            "edges_int": artifact["bands"]["edges_int"],
        },
        "calibration_mode": (artifact.get("calibration") or {}).get("mode"),
        "missing_policy": artifact["features"].get("missing_policy"),
        "monotone_constraints": artifact["model"].get("monotone_constraints"),
        "reason_coverage": meta.get("reason_coverage"),
        "model_family": meta.get("model_family"),
        "compileml_version": meta.get("compileml_version"),
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_verify(args) -> int:
    try:
        artifact = load_artifact(args.artifact)
    except (ArtifactError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"OK: hash verified ({artifact['artifact_hash'][:16]}…)")
    return 0


def cmd_score(args) -> int:
    artifact = load_artifact(args.artifact)
    names = artifact["features"]["names"]

    if args.features:
        values = [
            None if v.lower() in ("", "nan", "null") else float(v) for v in args.features.split(",")
        ]
        payload = decide(artifact, values, explain=args.explain, top_k=args.top_k)
        print(json.dumps(payload, indent=2))
        return 0

    if not args.csv:
        raise SystemExit("provide --features or --csv")
    rows, _ = _read_csv(args.csv, names)
    to_stdout = args.out in (None, "-")
    out = (
        sys.stdout
        if to_stdout
        else open(args.out, "w", newline="", encoding="utf-8")  # noqa: SIM115 — closed in finally
    )
    try:
        writer = csv.writer(out)
        writer.writerow(["row", "band", "latent_int", "pd"])
        for i, row in enumerate(rows):
            p = decide(artifact, row, explain=False)
            writer.writerow([i, p["band"], p["latent_int"], p["pd"]])
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def cmd_export(args) -> int:
    artifact = load_artifact(args.artifact)
    if args.target == "cobol":
        from compileml.export import export_cobol

        text = export_cobol(artifact, program_id=args.program_id)
    else:
        from compileml.export import export_sql

        text = export_sql(artifact, table=args.table, dialect=args.dialect)
    if args.out in (None, "-"):
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    return 0


def cmd_scorecard(args) -> int:
    from compileml.scorecard import build_scorecard, scorecard_to_csv, scorecard_to_markdown

    artifact = load_artifact(args.artifact)
    scorecard = build_scorecard(artifact)  # raises above depth 2 with guidance
    text = scorecard_to_csv(scorecard) if args.format == "csv" else scorecard_to_markdown(scorecard)
    if args.out in (None, "-"):
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    return 0


def cmd_validate(args) -> int:
    from compileml.validate import validate_artifact

    artifact = load_artifact(args.artifact)
    X_val = y_val = None
    if args.csv:
        names = artifact["features"]["names"]
        rows, labels = _read_csv(args.csv, names, y_col=args.y_col)
        baseline = artifact["features"]["baseline"]
        X_val = [[baseline[j] if v is None else v for j, v in enumerate(row)] for row in rows]
        y_val = labels if args.y_col else None
    report = validate_artifact(
        artifact,
        X_val=X_val,
        y_val=y_val,
        require_full_reason_coverage=args.require_reasons,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_pass"] else 1


def cmd_compile(args) -> int:
    import pickle

    import numpy as np

    from compileml.artifact import build_artifact, save_artifact
    from compileml.bands import monotone_quantile_bands, quantile_bands

    with open(args.model, "rb") as f:  # the user's own fitted model
        model = pickle.load(f)

    # Features = every CSV column except the outcome column.
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    feature_names = [c for c in header if c != args.y_col]
    rows, labels = _read_csv(args.csv, feature_names, y_col=args.y_col)

    X = np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)
    col_median = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), col_median, X)
    y = np.array(labels, dtype=int)

    latent = np.clip(np.asarray(model.predict(X), dtype=float), 0.0, 1.0)
    if args.y_col:
        spec = monotone_quantile_bands(latent, y, n_bands=args.n_bands)
    else:
        spec = quantile_bands(latent, n_bands=args.n_bands)

    reasons = None
    if args.reasons:
        with open(args.reasons, encoding="utf-8") as f:
            reasons = json.load(f)

    artifact = build_artifact(
        model,
        feature_names,
        col_median,
        spec,
        calibration_latent=latent if args.y_col else None,
        calibration_y=y if args.y_col else None,
        reasons=reasons,
        X_sample=X[: min(len(X), 500)],
    )
    save_artifact(artifact, args.out)
    print(f"wrote {args.out}  (hash {artifact['artifact_hash'][:16]}…)")
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compileml",
        description="Compile tree models into deterministic, auditable decision artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="summarize an artifact")
    p.add_argument("artifact")
    p.add_argument("--no-verify", action="store_true", help="skip hash verification")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("verify", help="verify an artifact's hash and structure")
    p.add_argument("artifact")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("score", help="score one row or a CSV through an artifact")
    p.add_argument("artifact")
    p.add_argument("--features", help="comma-separated values in artifact feature order")
    p.add_argument("--csv", help="CSV with feature columns named as in the artifact")
    p.add_argument("--out", help="output CSV path (default: stdout)")
    p.add_argument("--explain", action="store_true", help="include reasons (single row)")
    p.add_argument("--top-k", type=int, default=None)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("export", help="export the artifact to native code")
    p.add_argument("artifact")
    p.add_argument("--target", choices=["cobol", "sql"], required=True)
    p.add_argument("--out", help="output file (default: stdout)")
    p.add_argument("--table", default="features", help="SQL source table name")
    p.add_argument("--dialect", choices=["ansi", "sqlite"], default="ansi")
    p.add_argument("--program-id", default="CMLSCORE", help="COBOL PROGRAM-ID")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("scorecard", help="collapse a depth<=2 artifact into an exact scorecard")
    p.add_argument("artifact")
    p.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    p.add_argument("--out", help="output file (default: stdout)")
    p.set_defaults(func=cmd_scorecard)

    p = sub.add_parser("validate", help="run the 10-check validation framework")
    p.add_argument("artifact")
    p.add_argument("--csv", help="validation CSV (features + outcome)")
    p.add_argument("--y-col", help="outcome column name in --csv")
    p.add_argument(
        "--require-reasons", action="store_true", help="fail if reason coverage is below 100%%"
    )
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("compile", help="compile a pickled model + CSV into an artifact")
    p.add_argument("--model", required=True, help="pickled fitted model (.pkl)")
    p.add_argument("--csv", required=True, help="training/calibration CSV")
    p.add_argument("--y-col", help="outcome column (enables calibration + bad-rate bands)")
    p.add_argument("--n-bands", type=int, default=10)
    p.add_argument("--reasons", help="reason dictionary JSON file")
    p.add_argument("--out", required=True, help="output artifact path")
    p.set_defaults(func=cmd_compile)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
