"""CompileML benchmark suite.

Reproduces every performance and retention number stated in the README.
Fully deterministic: synthetic credit-style data with a fixed seed, no
network access. Run it yourself:

    python benchmarks/run_benchmarks.py

Results land in benchmarks/results.json and print as a markdown table.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import compileml
from compileml.artifact import build_artifact, save_artifact
from compileml.bands import monotone_quantile_bands, quantile_bands
from compileml.compile import train_whitebox
from compileml.runtime import LEAF, contributions_half_micro, decide, load_artifact
from compileml.runtime.bands import band_index
from compileml.runtime.explain import contributions_half_micro_reference

SEED = 42
HERE = Path(__file__).resolve().parent

# The explanation-cost sweep. Tree count matches the headline artifact so the
# sweep and the headline describe a model of the same size.
SWEEP_FEATURES = (8, 23, 50, 100)
SWEEP_TREES = 120
SWEEP_ROWS = 40


def make_credit_data(n: int, p: int, rng: np.random.Generator):
    """Synthetic credit-style features with interactions and a known DGP."""
    X = rng.standard_normal((n, p))
    logit = (
        1.3 * X[:, 0]  # utilization
        - 1.0 * X[:, 1]  # payment ratio
        + 0.8 * X[:, 2] * X[:, 3]  # interaction: balance x limit
        + 0.5 * X[:, 4]
        - 1.2
    )
    p_default = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p_default).astype(int)
    return X, y


def explain_cost_by_features() -> tuple[dict, dict]:
    """Attribution cost alone, at several feature counts, by both derivations.

    Times the decomposition itself rather than ``decide()``, so fixed per-row
    work — row preparation, banding, calibration, reason formatting — cannot
    flatten the curve and make the claim look better than it is.

    The target spreads signal over every feature, so at 100 features the
    trees genuinely split on many of them. Flatness here is not an artefact
    of a model that ignores most of its inputs; ``features_split_on`` records
    how many it actually uses.

    Milliseconds belong to one machine, so each entry also counts tree walks
    per row, which belong to the algorithm. The per-tree path walks each tree
    once per subset of the features that tree splits on, ``sum(2**k)`` —
    at most eight walks a tree at depth 2, whatever ``p`` is. The
    perturbation path scores the whole ensemble ``2 + p + p(p-1)/2`` times:
    the row, the baseline, each feature perturbed, each pair perturbed.
    Both counts were checked against instrumented runs.

    Both derivations must reach identical integers on every timed row. A
    timing is not reported for a path that disagrees with the other.
    """
    by_p: dict = {}
    for p in SWEEP_FEATURES:
        rng = np.random.default_rng(SEED + p)
        X = rng.standard_normal((4000, p))
        w = rng.choice([-1.0, 1.0], size=p) / np.sqrt(np.arange(1, p + 1))
        target = 1.0 / (1.0 + np.exp(-(X @ w)))

        whitebox, _ = train_whitebox(X, target, n_estimators=SWEEP_TREES, random_state=SEED)
        latent = np.clip(whitebox.predict(X), 0, 1)
        artifact = build_artifact(
            whitebox,
            [f"f{j:03d}" for j in range(p)],
            np.median(X, axis=0),
            quantile_bands(latent, n_bands=5),
        )
        model = artifact["model"]
        baseline = [float(b) for b in artifact["features"]["baseline"]]

        per_tree_ms, perturbation_ms = [], []
        for row in ([float(v) for v in r] for r in X[:SWEEP_ROWS]):
            t0 = time.perf_counter()
            fast = contributions_half_micro(model, row, baseline)
            per_tree_ms.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            reference = contributions_half_micro_reference(model, row, baseline)
            perturbation_ms.append((time.perf_counter() - t0) * 1000)

            if fast != reference:
                raise AssertionError(f"attribution derivations disagree at p={p}")

        split_sets = [{f for f in tree["feature"] if f != LEAF} for tree in model["trees"]]
        by_p[str(p)] = {
            "per_tree_ms": round(statistics.median(per_tree_ms), 3),
            "perturbation_ms": round(statistics.median(perturbation_ms), 3),
            "per_tree_walks": sum(2 ** len(fs) for fs in split_sets),
            "perturbation_walks": (2 + p + p * (p - 1) // 2) * len(split_sets),
            "features_split_on": len(set().union(*split_sets)),
        }
        print(f"  p={p:>3}: {by_p[str(p)]}")

    config = {
        "n_trees": SWEEP_TREES,
        "max_depth": 2,
        "rows_timed": SWEEP_ROWS,
        "measures": "attribution only, per row: median ms, and tree walks",
        "derivations_agree": True,  # reaching this line requires it
    }
    return by_p, config


def median_p95(samples_ms: list[float]) -> tuple[float, float]:
    return (
        statistics.median(samples_ms),
        statistics.quantiles(samples_ms, n=20)[18],  # p95
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    results: dict = {
        "seed": SEED,
        # The version measured, so a benchmark run against a stale installed
        # copy cannot pass for a measurement of the checked-out source.
        "compileml": compileml.__version__,
        "python": platform.python_version(),
        "machine": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
    }

    # ------------------------------------------------------------ retention
    n, p = 40_000, 23
    X, y = make_credit_data(n, p, rng)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    feature_names = [f"feature_{i:02d}" for i in range(p)]

    print("training teacher (sklearn GBM, 300 trees, depth 4)…")
    teacher = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED
    ).fit(X_train, y_train)
    teacher_latent_train = teacher.predict_proba(X_train)[:, 1]
    teacher_latent_test = teacher.predict_proba(X_test)[:, 1]

    print("distilling whitebox (120 trees, depth 2)…")
    whitebox, fidelity = train_whitebox(
        X_train, teacher_latent_train, n_estimators=120, random_state=SEED
    )
    latent_train = np.clip(whitebox.predict(X_train), 0, 1)

    bands = monotone_quantile_bands(latent_train, y_train, n_bands=10)
    artifact = build_artifact(
        whitebox,
        feature_names,
        np.median(X_train, axis=0),
        bands,
        calibration_latent=latent_train,
        calibration_y=y_train,
        X_sample=X_train[:500],
    )
    path = HERE / "benchmark_artifact.json"
    save_artifact(artifact, path)
    artifact = load_artifact(path)

    rows_test = [[float(v) for v in row] for row in X_test]
    decisions = [decide(artifact, row, explain=False) for row in rows_test]
    artifact_latent = np.array([d["raw_micro"] for d in decisions], dtype=float) / 1e6
    artifact_band = np.array([d["band_idx"] for d in decisions], dtype=float)

    def gini(score) -> float:
        return 2 * roc_auc_score(y_test, score) - 1

    g_teacher = gini(teacher_latent_test)
    g_artifact = gini(artifact_latent)
    g_band = gini(artifact_band)
    from scipy.stats import spearmanr

    results["retention"] = {
        "teacher_auc": round(roc_auc_score(y_test, teacher_latent_test), 4),
        "artifact_auc": round(roc_auc_score(y_test, artifact_latent), 4),
        "teacher_gini": round(g_teacher, 4),
        "artifact_gini": round(g_artifact, 4),
        "band_ordinal_gini": round(g_band, 4),
        "gini_retention_pct": round(100 * g_artifact / g_teacher, 2),
        "band_gini_retention_pct": round(100 * g_band / g_teacher, 2),
        "spearman_teacher_vs_artifact": round(
            float(spearmanr(teacher_latent_test, artifact_latent)[0]), 4
        ),
        "distill_spearman_train": round(fidelity["spearman"], 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": p,
    }

    # ------------------------------------------------------------ latency
    print("measuring latency…")
    reps = 400
    score_ms = []
    for i in range(reps):
        row = rows_test[i % len(rows_test)]
        t0 = time.perf_counter()
        decide(artifact, row, explain=False)
        score_ms.append((time.perf_counter() - t0) * 1000)

    explain_ms = []
    for i in range(60):
        row = rows_test[i % len(rows_test)]
        t0 = time.perf_counter()
        decide(artifact, row, explain=True)
        explain_ms.append((time.perf_counter() - t0) * 1000)

    edges = artifact["bands"]["edges_int"]
    band_us = []
    for i in range(reps):
        latent_int = decisions[i % len(decisions)]["latent_int"]
        t0 = time.perf_counter()
        band_index(latent_int, edges)
        band_us.append((time.perf_counter() - t0) * 1_000_000)

    print("measuring explanation cost by feature count…")
    by_features, by_features_config = explain_cost_by_features()

    n_trees = len(artifact["model"]["trees"])
    s_med, s_p95 = median_p95(score_ms)
    e_med, e_p95 = median_p95(explain_ms)
    results["latency"] = {
        "score_band_pd_median_ms": round(s_med, 3),
        "score_band_pd_p95_ms": round(s_p95, 3),
        "full_explain_median_ms": round(e_med, 3),
        "full_explain_p95_ms": round(e_p95, 3),
        "band_ladder_median_us": round(statistics.median(band_us), 2),
        "explain_by_features": by_features,
        "explain_by_features_config": by_features_config,
        "note": (
            f"pure-Python stdlib runtime, single row, {n_trees} depth-2 trees at "
            f"p={p}. full_explain is the whole decide() payload: score, band, PD, "
            "exact attribution and reason codes. Attribution is aggregated per "
            "tree, so its cost is O(trees) and independent of feature count; "
            "explain_by_features times the attribution alone by both derivations, "
            "with tree walks per row as the machine-independent measure."
        ),
    }

    # ------------------------------------------------------------ artifact
    results["artifact"] = {
        "size_kb": round(path.stat().st_size / 1024, 1),
        "n_trees": len(artifact["model"]["trees"]),
        "hash": artifact["artifact_hash"][:16] + "…",
        "exact_attribution": artifact["runtime"]["exact_attribution"],
    }

    # ------------------------------------------------------- determinism
    print("checking build determinism…")
    artifact2 = build_artifact(
        whitebox,
        feature_names,
        np.median(X_train, axis=0),
        monotone_quantile_bands(latent_train, y_train, n_bands=10),
        calibration_latent=latent_train,
        calibration_y=y_train,
        X_sample=X_train[:500],
    )
    results["determinism"] = {
        "rebuild_hash_identical": artifact2["artifact_hash"] == artifact["artifact_hash"],
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}\n")

    r, lat, art = results["retention"], results["latency"], results["artifact"]
    rows = [
        ("Teacher Gini", f"{r['teacher_gini']}"),
        ("Artifact Gini (integer)", f"{r['artifact_gini']} ({r['gini_retention_pct']}% retention)"),
        (
            "Band-ordinal Gini",
            f"{r['band_ordinal_gini']} ({r['band_gini_retention_pct']}% retention)",
        ),
        ("Spearman teacher vs artifact", f"{r['spearman_teacher_vs_artifact']}"),
        (
            "score+band+PD latency (median / p95)",
            f"{lat['score_band_pd_median_ms']} / {lat['score_band_pd_p95_ms']} ms",
        ),
        (
            "full explained decision (median / p95)",
            f"{lat['full_explain_median_ms']} / {lat['full_explain_p95_ms']} ms",
        ),
        ("band ladder only", f"{lat['band_ladder_median_us']} µs"),
        ("artifact size", f"{art['size_kb']} KB"),
        ("rebuild hash identical", f"{results['determinism']['rebuild_hash_identical']}"),
    ]
    print("| Metric | Value |")
    print("|---|---|")
    for name, value in rows:
        print(f"| {name} | {value} |")

    print(f"\nattribution alone, {SWEEP_TREES} depth-2 trees (median ms per row)\n")
    print("| features | perturbation ms | per tree ms | perturbation walks | per tree walks |")
    print("|---|---|---|---|---|")
    for feats, v in lat["explain_by_features"].items():
        print(
            f"| {feats} | {v['perturbation_ms']} | {v['per_tree_ms']} "
            f"| {v['perturbation_walks']:,} | {v['per_tree_walks']:,} |"
        )


if __name__ == "__main__":
    main()
