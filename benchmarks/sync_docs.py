"""Copy the committed benchmark figures into the documents that transcribe them.

benchmarks/results.json is generated; the README table, the attribution page's
cost table and two sentences in the docs are transcribed from it. A transcribed
number that has drifted from the file is worse than no number, so this script
is the only way those copies get written. Run it after run_benchmarks.py:

    python benchmarks/run_benchmarks.py
    python benchmarks/sync_docs.py

It is idempotent, and it refuses to guess: every target is matched exactly
once or the script stops.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results.json"
README = ROOT / "README.md"
ATTRIBUTION = ROOT / "docs" / "concepts" / "attribution.md"
FAQ = ROOT / "docs" / "faq.md"

SWEEP_ROW = r"^\| {p} \| [0-9.]+ ms \| [0-9.]+ ms \| [0-9,]+ \| [0-9,]+ \|$"


def read(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def replace_once(text: str, pattern: str, replacement: str, where: str, flags=re.M) -> str:
    matches = re.findall(pattern, text, flags)
    if len(matches) != 1:
        raise SystemExit(
            f"{where}: expected exactly one match for {pattern!r}, found {len(matches)}"
        )
    return re.sub(pattern, lambda _: replacement, text, count=1, flags=flags)


def readme_table(res: dict) -> str:
    r, lat, art, det = res["retention"], res["latency"], res["artifact"], res["determinism"]
    lo, hi = r["retention_ci"]
    retained = f"{r['gini_retention_pct']:.1f}% retained ({lo:.1f}–{hi:.1f})"
    rows = [
        ("Ceiling Gini, 300-tree GBM", f"{r['teacher_gini']:.3f}"),
        ("**Compiled integer artifact Gini**", f"**{r['artifact_gini']:.3f} — {retained}**"),
        (
            "Floor Gini, WoE logistic regression",
            f"{r['floor_gini']:.3f} — artifact at {r['floor_ratio_pct']:.1f}%",
        ),
        (
            "Band-ordinal Gini, 10 bands",
            f"{r['band_ordinal_gini']:.3f} — {r['band_gini_retention_pct']:.1f}% retained",
        ),
        ("Spearman correlation, ceiling vs. artifact", f"{r['spearman_teacher_vs_artifact']:.3f}"),
        (
            f"Selected on Select, of {r['n_configs']} configurations",
            f"α = {r['selected_alpha']}, {r['selected_trees']} trees, depth {r['selected_depth']}",
        ),
        ("Score + band + calibrated PD", f"**{lat['score_band_pd_median_ms']:.2f} ms median**"),
        ("Score + band + calibrated PD, p95", f"{lat['score_band_pd_p95_ms']:.2f} ms"),
        (
            f"Full explained decision, {art['n_trees']} trees",
            f"{lat['full_explain_median_ms']:.2f} ms median",
        ),
        ("Full explained decision, p95", f"{lat['full_explain_p95_ms']:.2f} ms"),
        ("Band assignment alone", f"{lat['band_ladder_median_us']:.1f} µs"),
        ("Artifact size", f"{art['size_kb']:.0f} KB"),
        (
            "Same configuration and identical hash on rerun",
            "Yes" if det["rebuild_hash_identical"] and det["same_configuration_selected"] else "No",
        ),
    ]
    w = max(len(a) for a, _ in rows)
    v = max(len(b) for _, b in rows)
    lines = [f"| {'Metric':<{w}} | {'Value':>{v}} |", f"| {'-' * w} | {'-' * (v - 1)}: |"]
    lines += [f"| {a:<{w}} | {b:>{v}} |" for a, b in rows]
    return "\n".join(lines) + "\n"


def sweep_rows(text: str, sweep: dict, where: str) -> str:
    for p in ("8", "23", "50", "100"):
        v = sweep[p]
        row = (
            f"| {p} | {v['perturbation_ms']:.2f} ms | {v['per_tree_ms']:.2f} ms | "
            f"{v['perturbation_walks']:,} | {v['per_tree_walks']:,} |"
        )
        text = replace_once(text, SWEEP_ROW.format(p=p), row, f"{where} sweep row p={p}")
    return text


def main() -> None:
    res = json.loads(read(RESULTS))
    lat = res["latency"]
    sweep = lat["explain_by_features"]
    per_tree = [sweep[p]["per_tree_ms"] for p in ("8", "23", "50", "100")]

    readme = read(README)
    readme = replace_once(
        readme, r"\| Metric[^\n]*\n\|[- |:]+\n(?:\|[^\n]*\n)+", readme_table(res), "README table"
    )
    readme = sweep_rows(readme, sweep, "README")
    write(README, readme)

    attribution = sweep_rows(read(ATTRIBUTION), sweep, "attribution.md")
    attribution = replace_once(
        attribution,
        r"At [0-9.]+ ms it is under",
        f"At {lat['full_explain_median_ms']:.2f} ms it is under",
        "attribution.md batch sentence",
    )
    write(ATTRIBUTION, attribution)

    faq = replace_once(
        read(FAQ),
        r"[0-9.]+–[0-9.]+ ms per row whether the model has",
        f"{min(per_tree):.1f}–{max(per_tree):.1f} ms per row whether the model has",
        "faq.md cost range",
    )
    write(FAQ, faq)
    print(
        "synced README.md, docs/concepts/attribution.md, docs/faq.md from benchmarks/results.json"
    )


if __name__ == "__main__":
    main()
