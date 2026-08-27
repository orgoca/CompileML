"""Configuration sweeps: measure the knobs instead of guessing them.

Both sweeps return plain lists of dicts (tidy rows — feed them straight to
``pandas.DataFrame`` if you like). Nothing here changes an artifact; these
are compile-time design tools.

The asymmetry worth internalizing before sweeping:

- ``n_estimators`` buys fidelity at a **linear** cost in artifact size and
  explanation time, and costs nothing else. Determinism, portability, and
  exact attribution are unaffected at any tree count.
- ``max_depth`` buys fidelity per tree but **takes the exactness guarantee
  above 2** (attribution residuals appear, and no clean scorecard exists).

Spend on trees; be stingy with depth.
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from compileml.bands.builders import monotone_quantile_bands
from compileml.bands.efficiency import band_efficiency
from compileml.compile.distill import train_whitebox
from compileml.compile.extract import extract_trees
from compileml.compile.quantize import quantize_model
from compileml.runtime.explain import contributions_half_micro


def sweep_whitebox(
    X,
    teacher_latent,
    y,
    *,
    trees_grid=(10, 20, 40, 80, 160),
    depth_grid=(1, 2, 3),
    learning_rate: float = 0.2,
    random_state: int = 42,
    X_val=None,
    y_val=None,
    teacher_latent_val=None,
    monotone_constraints=None,
    explain_timing_rows: int = 3,
) -> list[dict]:
    """Grid-sweep whitebox capacity; measure what each configuration buys.

    Per configuration: Gini and retention vs the teacher, Spearman rank
    agreement with the teacher, whether attribution is exact
    (``depth <= 2``), the quantized model's JSON size, and a measured
    per-row exact-explanation cost.

    Supply a holdout (``X_val`` / ``y_val`` / ``teacher_latent_val``) for
    honest numbers; in-sample retention flatters every configuration.

    Pass ``monotone_constraints`` to sweep the constrained backend instead —
    run both and diff the retention column to measure the monotonicity
    premium before committing to it.
    """
    X_arr = np.asarray(X, dtype=float)
    t_lat = np.asarray(teacher_latent, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=int).reshape(-1)

    if X_val is not None:
        X_eval = np.asarray(X_val, dtype=float)
        y_eval = np.asarray(y_val, dtype=int).reshape(-1)
        t_eval = np.asarray(teacher_latent_val, dtype=float).reshape(-1)
        in_sample = False
    else:
        X_eval, y_eval, t_eval = X_arr, y_arr, t_lat
        in_sample = True

    teacher_gini = 2 * float(roc_auc_score(y_eval, t_eval)) - 1
    baseline = np.median(X_arr, axis=0)

    rows = []
    for depth in depth_grid:
        for n_trees in trees_grid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # depth>2 warning is the sweep's point
                model, _ = train_whitebox(
                    X_arr,
                    t_lat,
                    n_estimators=n_trees,
                    max_depth=depth,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    monotone_constraints=monotone_constraints,
                )
            latent_eval = np.clip(model.predict(X_eval), 0.0, 1.0)
            gini = 2 * float(roc_auc_score(y_eval, latent_eval)) - 1
            rho = float(spearmanr(t_eval, latent_eval)[0])

            model_int = quantize_model(extract_trees(model))
            model_kb = len(json.dumps(model_int)) / 1024

            times = []
            base_list = [float(v) for v in baseline]
            for i in range(min(explain_timing_rows, len(X_eval))):
                row = [float(v) for v in X_eval[i]]
                t0 = time.perf_counter()
                contributions_half_micro(model_int, row, base_list)
                times.append((time.perf_counter() - t0) * 1000)

            rows.append(
                {
                    "n_estimators": int(n_trees),
                    "max_depth": int(depth),
                    "gini": round(gini, 4),
                    "gini_retention_pct": (
                        round(100 * gini / teacher_gini, 2) if teacher_gini > 0 else None
                    ),
                    "spearman_vs_teacher": round(rho, 4),
                    "exact_attribution": depth <= 2,
                    "model_kb": round(model_kb, 1),
                    "explain_ms_per_row": round(float(np.median(times)), 2),
                    "constrained": monotone_constraints is not None,
                    "in_sample": in_sample,
                }
            )
    return rows


def sweep_bands(
    latent,
    y,
    *,
    k_grid=(4, 6, 8, 10, 12, 16),
    scale: int = 1000,
    n_boot: int = 100,
    seed: int = 7,
) -> list[dict]:
    """Sweep fixed-K band counts; measure what each ladder costs and keeps.

    Per K: band-ordinal Gini and its retention of the continuous latent's
    Gini, the Gini gap ("money on the table"), the worst within-band AUC
    with its refinement verdict, the smallest band's volume, and any
    integer-edge collisions at the display scale (a K too fine for the
    scale to represent).

    For *discovering* K instead of sweeping it, see ``semantic_bands`` and
    ``governance_bands`` — they return the number of bands the data can
    statistically defend.
    """
    F = np.asarray(latent, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=int).reshape(-1)

    rows = []
    for k in k_grid:
        spec = monotone_quantile_bands(F, y_arr, n_bands=int(k))
        from compileml.compile.quantize import rha

        edges_int = [rha(float(e) * scale) for e in spec.edges]
        collisions = sum(1 for a, b in zip(edges_int, edges_int[1:]) if b <= a)

        eff = band_efficiency(F, y_arr, spec, scale=scale, n_boot=n_boot, seed=seed)
        occupied = [p for p in eff["per_band"] if p["n"] > 0]
        worst = max(
            (p for p in occupied if p["auc_ci"] is not None),
            key=lambda p: p["auc_ci"][0],
            default=None,
        )
        rates = [p["bad_rate"] for p in occupied]
        violations = sum(1 for a, b in zip(rates, rates[1:]) if b < a - 0.01)

        rows.append(
            {
                "n_bands": int(k),
                "band_ordinal_gini": eff["band_ordinal_gini"],
                "gini_retention_pct": (
                    round(100 * eff["band_ordinal_gini"] / eff["continuous_gini"], 2)
                    if eff["continuous_gini"] > 0
                    else None
                ),
                "gini_gap": eff["gini_gap"],
                "worst_within_band_auc": worst["auc_within"] if worst else None,
                "worst_band_verdict": worst["verdict"] if worst else None,
                "min_band_n": min((p["n"] for p in occupied), default=0),
                "monotonicity_violations": int(violations),
                "int_edge_collisions": int(collisions),
            }
        )
    return rows
