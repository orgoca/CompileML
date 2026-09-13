"""Stability monitoring: the part only a compiled artifact can do.

Three questions a risk committee asks every period:

- **Which features moved the score?** :func:`drift_decomposition` splits the
  change in mean raw score between two populations into per-feature shifts
  that sum to it exactly.
- **Are the bands still calibrated?** :func:`band_drift` compares observed bad
  rates against the PD the artifact emitted, per logged band.
- **Does the baseline still describe anyone?** :func:`baseline_staleness`
  measures how far the frozen baseline has moved in population rank.

This is learning-side code: it reads decisions after they were made and never
makes one, so it is free to use NumPy. It deliberately implements no PSI, CSI
or distribution tests — the tools you already run do those well. See
``docs/howto/monitor.md``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from compileml._decomposition import mean_gap_by_feature, require_exact_contributions
from compileml.fairness.metrics import wilson_interval
from compileml.runtime.calibrate import PD_SCALE


def drift_decomposition(reference_decisions, current_decisions) -> dict:
    """Decompose the change in mean raw score, current minus reference, by feature.

    Both lists come from ``decide(artifact, row, include_contributions=True)``
    under one artifact. The quantity decomposed is mean raw score movement,
    ``2 * (raw_micro - baseline_micro)`` in half-micro units; both populations
    share the baseline, so it is exactly the change in mean raw score. The
    per-feature shifts sum to it with ``==`` — the same arithmetic as fairness
    §6, and the same report shape.

    ``latent_int`` is not decomposed, because the runtime clamps it to the
    unit interval; each side's mean is reported under ``latent_context`` for
    reading alongside.

    After ``recalibrate_artifact`` the artifact hash changes but the model and
    baseline do not. Re-run ``decide()`` on the reference rows under the new
    artifact; their contributions come out identical.

    Raises:
        ValueError: if either list is empty, lacks contributions, carries a
            nonzero attribution residual, or the decisions come from more
            than one artifact.
    """
    require_exact_contributions(reference_decisions, "drift_decomposition")
    require_exact_contributions(current_decisions, "drift_decomposition")

    hashes = {d.get("artifact_hash") for d in [*reference_decisions, *current_decisions]}
    if len(hashes) != 1:
        raise ValueError(
            f"the decisions come from {len(hashes)} artifacts; drift is only "
            "defined under one. If the artifact was recalibrated, re-run "
            "decide() on the reference rows under the current artifact — the "
            "model and baseline are unchanged, so the contributions will be too."
        )

    contributions = sorted(current_decisions[0]["contributions"], key=lambda c: c["index"])
    feature_names = [c.get("feature", str(c["index"])) for c in contributions]
    n_ref, n_cur = len(reference_decisions), len(current_decisions)

    latent_ref = Fraction(sum(int(d["latent_int"]) for d in reference_decisions), n_ref)
    latent_cur = Fraction(sum(int(d["latent_int"]) for d in current_decisions), n_cur)

    return {
        "artifact_hash": hashes.pop(),
        "direction": "current minus reference",
        "reference_n": n_ref,
        "current_n": n_cur,
        **mean_gap_by_feature(current_decisions, reference_decisions, feature_names),
        "latent_context": {
            "reference_mean_latent_int": float(latent_ref),
            "current_mean_latent_int": float(latent_cur),
            "shift": float(latent_cur - latent_ref),
        },
    }


def band_drift(artifact, decisions, y, *, min_n: int = 30, z: float = 1.96) -> dict:
    """Observed bad rate against the emitted PD, per logged band.

    Rows are grouped by the band each decision logged, never by a band rebuilt
    from scores, so a row on a boundary is counted where production put it.
    Every band in the artifact is listed, including empty ones.

    ``outside_interval`` is whether the band's mean emitted PD falls outside
    the Wilson interval on its observed bad rate — ``None`` below ``min_n``,
    where a flag would mean little. At 95% intervals a perfectly calibrated
    artifact still flags about one band in twenty; with ten bands, expect one
    flag every two periods on average.

    Outcomes must have matured: last month's decisions have not had time to
    default, and will look better calibrated than they are.

    Nothing is recalibrated. The report is evidence for an explicit call to
    ``recalibrate_artifact``.
    """
    artifact_hash = artifact.get("artifact_hash")
    stray = sum(1 for d in decisions if d.get("artifact_hash") != artifact_hash)
    if stray:
        raise ValueError(
            f"{stray} of {len(decisions)} decisions were not made by this artifact "
            "(their artifact_hash differs)"
        )
    outcomes = np.asarray(y).reshape(-1)
    if outcomes.size != len(decisions):
        raise ValueError("decisions and y must have the same length")
    if not np.isin(outcomes, [0, 1]).all():
        raise ValueError("y must contain only 0 and 1 outcomes")

    labels = list(artifact["bands"]["labels"])
    n = dict.fromkeys(labels, 0)
    bad = dict.fromkeys(labels, 0)
    pd_ppm_sum = dict.fromkeys(labels, 0)
    for d, outcome in zip(decisions, outcomes.tolist()):
        band = d["band"]
        if band not in n:
            raise ValueError(f"decision band {band!r} is not one of the artifact's bands")
        n[band] += 1
        bad[band] += int(outcome)
        pd_ppm_sum[band] += int(d["pd_ppm"])

    bands = []
    for label in labels:
        if n[label] == 0:
            bands.append(
                {
                    "band": label,
                    "n": 0,
                    "bad": 0,
                    "observed_bad_rate": None,
                    "observed_ci": None,
                    "mean_pd": None,
                    "outside_interval": None,
                }
            )
            continue
        lo, hi = wilson_interval(bad[label], n[label], z=z)
        mean_pd = float(Fraction(pd_ppm_sum[label], n[label] * PD_SCALE))
        bands.append(
            {
                "band": label,
                "n": n[label],
                "bad": bad[label],
                "observed_bad_rate": bad[label] / n[label],
                "observed_ci": [lo, hi],
                "mean_pd": mean_pd,
                "outside_interval": None if n[label] < min_n else not lo <= mean_pd <= hi,
            }
        )
    return {"artifact_hash": artifact_hash, "z": z, "min_n": min_n, "bands": bands}


def _baseline_percentile(column: np.ndarray, baseline: float) -> tuple[float, float]:
    """Where the baseline sits in a column, midpoint for ties; and the missing rate."""
    present = column[~np.isnan(column)]
    missing_rate = 1.0 - present.size / column.size if column.size else float("nan")
    if present.size == 0:
        return float("nan"), missing_rate
    below = np.count_nonzero(present < baseline)
    tied = np.count_nonzero(present == baseline)
    return float((below + 0.5 * tied) / present.size), missing_rate


def baseline_staleness(artifact, X_reference, X_current) -> dict:
    """How far the frozen baseline has moved in population rank, per feature.

    The baseline is both the imputation value and the attribution reference
    point, so it matters when it stops describing the population. Raw deltas
    are not comparable across features, so this measures rank: the
    baseline's percentile among non-missing values in each population, with
    ties counted half, and the shift between them. Missing rates are reported
    alongside, because a missing value is imputed at the baseline.

    Features are ranked by absolute percentile shift. No pass/fail threshold
    is applied.
    """
    names = list(artifact["features"]["names"])
    baseline = [float(b) for b in artifact["features"]["baseline"]]
    matrices = []
    for label, X in (("X_reference", X_reference), ("X_current", X_current)):
        M = np.asarray(X, dtype=float)
        if M.ndim != 2 or M.shape[1] != len(names):
            raise ValueError(f"{label} must be a matrix with {len(names)} feature columns")
        matrices.append(M)
    ref, cur = matrices

    rows: list[dict] = []
    for j, name in enumerate(names):
        q_ref, missing_ref = _baseline_percentile(ref[:, j], baseline[j])
        q_cur, missing_cur = _baseline_percentile(cur[:, j], baseline[j])
        rows.append(
            {
                "feature": str(name),
                "baseline": baseline[j],
                "reference_percentile": q_ref,
                "current_percentile": q_cur,
                "percentile_shift": q_cur - q_ref,
                "reference_missing_rate": missing_ref,
                "current_missing_rate": missing_cur,
                "missing_rate_shift": missing_cur - missing_ref,
            }
        )
    rows.sort(key=lambda r: (np.isnan(r["percentile_shift"]), -abs(r["percentile_shift"])))
    return {"artifact_hash": artifact.get("artifact_hash"), "features": rows}
