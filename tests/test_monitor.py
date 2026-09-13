"""Monitoring: drift decomposes exactly, bands are read as logged, baselines by rank.

The acceptance criteria of #9, one or two tests each.
"""

import copy
import math
import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact
from compileml.bands import quantile_bands
from compileml.compile import train_whitebox
from compileml.fairness import attribution_disparity
from compileml.monitor import band_drift, baseline_staleness, drift_decomposition
from compileml.runtime import decide

N, P = 3000, 5
FEATURES = [f"f{i}" for i in range(P)]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((N, P))
    teacher = 1 / (1 + np.exp(-(1.1 * X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3] - 1.0)))
    y = (rng.random(N) < teacher).astype(int)
    return X, y, teacher


def _artifact(X, y, teacher, *, max_depth=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = train_whitebox(X, teacher, n_estimators=25, max_depth=max_depth, random_state=0)
        latent = np.clip(model.predict(X), 0, 1)
        return build_artifact(
            model,
            FEATURES,
            np.median(X, axis=0),
            quantile_bands(latent, n_bands=6),
            calibration_latent=latent,
            calibration_y=y,
        )


def _decide(artifact, X, **kw):
    return [decide(artifact, [float(v) for v in row], **kw) for row in X]


@pytest.fixture(scope="module")
def artifact(data):
    return _artifact(*data)


@pytest.fixture(scope="module")
def decisions(data, artifact):
    return _decide(artifact, data[0], include_contributions=True)


# --------------------------------------------------------- shared arithmetic
def test_drift_is_the_fairness_decomposition(data, decisions):
    """One implementation: the same split gives the same numbers either way."""
    g = (data[0][:, 4] > 0).astype(int)
    fairness = attribution_disparity(decisions, g, FEATURES)  # group 0 minus group 1
    drift = drift_decomposition(
        [d for d, k in zip(decisions, g) if k == 1],  # reference
        [d for d, k in zip(decisions, g) if k == 0],  # current
    )
    for key in ("mean_gap_half_micro", "sum_of_feature_gaps", "residual", "by_feature"):
        assert drift[key] == fairness[key]


def _integer_decisions(seed: int, n: int = 60, p: int = 7):
    """Exact hand-built decisions on which float means leave a residual (see test_fairness)."""
    rng = np.random.default_rng(seed)
    contrib = rng.integers(-900_001, 900_001, size=(n, p), dtype=np.int64)
    contrib[:, 0] += contrib.sum(1) % 2
    g = (rng.random(n) < 0.37).astype(int)
    decisions = [
        {
            "artifact_hash": "h",
            "contributions": [
                {"index": j, "feature": f"f{j}", "impact_half_micro": int(row[j])} for j in range(p)
            ],
            "attribution_residual_half_micro": 0,
            "raw_micro": int(row.sum()) // 2,
            "baseline_micro": 0,
            "latent_int": 500,
        }
        for row in contrib
    ]
    return decisions, g


def test_residual_is_exactly_zero_where_float_means_are_not():
    decisions, g = _integer_decisions(seed=2)
    reference = [d for d, k in zip(decisions, g) if k == 0]
    current = [d for d, k in zip(decisions, g) if k == 1]
    report = drift_decomposition(reference, current)
    assert report["residual"] == 0.0
    assert math.copysign(1.0, report["residual"]) == 1.0
    assert report["mean_gap_half_micro"] == report["sum_of_feature_gaps"]


def test_row_order_does_not_change_a_single_bit(decisions):
    rng = np.random.default_rng(4)
    reference, current = decisions[:1500], decisions[1500:]
    shuffled = drift_decomposition(
        [reference[i] for i in rng.permutation(len(reference))],
        [current[i] for i in rng.permutation(len(current))],
    )
    assert shuffled == drift_decomposition(reference, current)


# ------------------------------------------------------------ what it finds
def test_identical_populations_have_no_drift(decisions):
    report = drift_decomposition(decisions, decisions)
    assert report["mean_gap_half_micro"] == 0.0
    assert all(r["gap_half_micro"] == 0.0 for r in report["by_feature"])
    assert report["latent_context"]["shift"] == 0.0


def test_drift_planted_in_one_feature_is_attributed_to_it(data):
    """At depth 1 a feature's contribution depends on that feature alone, so a
    shift in one input moves exactly one contribution — the rest stay at 0.0."""
    X, y, teacher = data
    art = _artifact(X, y, teacher, max_depth=1)
    moved = X.copy()
    moved[:, 0] += 1.0
    report = drift_decomposition(
        _decide(art, X[:800], include_contributions=True),
        _decide(art, moved[:800], include_contributions=True),
    )
    top, *rest = report["by_feature"]
    assert top["feature"] == "f0"
    assert top["share_pct"] == pytest.approx(100.0)
    assert all(r["gap_half_micro"] == 0.0 for r in rest)
    assert report["residual"] == 0.0


def test_clamped_latent_is_context_not_decomposition():
    """Raw movement decomposes exactly even where the deployed latent is clamped."""

    def row(raw, latent):
        return {
            "artifact_hash": "h",
            "contributions": [{"index": 0, "feature": "f0", "impact_half_micro": 2 * raw}],
            "attribution_residual_half_micro": 0,
            "raw_micro": raw,
            "baseline_micro": 0,
            "latent_int": latent,
        }

    reference = [row(400_000, 400), row(600_000, 600)]
    current = [row(1_400_000, 1000), row(1_600_000, 1000)]  # latent clamped at 1000
    report = drift_decomposition(reference, current)
    assert report["mean_gap_half_micro"] == 2 * 1_000_000  # exact raw shift, half-micro
    assert report["residual"] == 0.0
    assert report["latent_context"]["shift"] == 500.0  # the clamp hides most of it


# ----------------------------------------------------------------- refusals
def test_refuses_decisions_without_contributions(data, artifact):
    plain = _decide(artifact, data[0][:50])
    with pytest.raises(ValueError, match="include_contributions"):
        drift_decomposition(plain, plain)


def test_refuses_inexact_attribution(data):
    X, y, teacher = data
    deep = _artifact(X, y, teacher, max_depth=3)
    dec = _decide(deep, X[:300], include_contributions=True)
    with pytest.raises(ValueError, match="residual"):
        drift_decomposition(dec[:150], dec[150:])


def test_refuses_decisions_from_two_artifacts(decisions):
    other = copy.deepcopy(decisions[:100])
    for d in other:
        d["artifact_hash"] = "0" * 64
    with pytest.raises(ValueError, match="re-run"):
        drift_decomposition(decisions[:100], other)


# ---------------------------------------------------------------- band drift
def test_band_drift_counts_the_logged_band_not_a_rebuilt_one(data, artifact, decisions):
    """Relabel every decision to one band: that is where band_drift must count them."""
    logged_elsewhere = [dict(d, band="G02") for d in decisions]
    report = band_drift(artifact, logged_elsewhere, data[1])
    counts = {b["band"]: b["n"] for b in report["bands"]}
    assert counts["G02"] == N
    assert sum(counts.values()) == N
    empty = [b for b in report["bands"] if b["n"] == 0]
    assert empty and all(b["outside_interval"] is None for b in empty)


def test_band_drift_flags_miscalibration_but_not_small_bands(data, artifact, decisions):
    all_bad = np.ones(N, dtype=int)  # every account defaulted: far above any emitted PD
    report = band_drift(artifact, decisions, all_bad)
    sized = [b for b in report["bands"] if b["n"] >= 30]
    assert sized, "fixture must have bands large enough to flag"
    assert all(b["outside_interval"] is True for b in sized)

    small = band_drift(artifact, decisions, all_bad, min_n=N + 1)
    assert all(b["outside_interval"] is None for b in small["bands"])


def test_band_drift_rarely_flags_a_calibrated_artifact(artifact, decisions):
    rng = np.random.default_rng(12)
    y_calibrated = (rng.random(N) < np.array([d["pd_ppm"] for d in decisions]) / 1e6).astype(int)
    report = band_drift(artifact, decisions, y_calibrated)
    flags = [b["outside_interval"] for b in report["bands"] if b["outside_interval"] is not None]
    assert sum(flags) <= 1


def test_band_drift_refuses_foreign_decisions_and_non_binary_outcomes(data, artifact, decisions):
    foreign = [dict(d, artifact_hash="0" * 64) for d in decisions[:10]]
    with pytest.raises(ValueError, match="not made by this artifact"):
        band_drift(artifact, foreign, data[1][:10])
    with pytest.raises(ValueError, match="only 0 and 1"):
        band_drift(artifact, decisions[:10], np.full(10, 0.5))


# ------------------------------------------------------- baseline staleness
def test_baseline_staleness_is_quiet_for_an_unchanged_population(data, artifact):
    X = data[0]
    rng = np.random.default_rng(21)
    fresh = rng.standard_normal((N, P))  # same distribution, different rows
    report = baseline_staleness(artifact, X, fresh)
    assert all(abs(r["percentile_shift"]) < 0.05 for r in report["features"])


def test_baseline_staleness_ranks_the_moved_feature_first_and_reports_missingness(data, artifact):
    X = data[0]
    current = X.copy()
    current[:, 3] += 1.5  # the baseline now sits low in this feature's distribution
    current[: N // 5, 1] = np.nan
    report = baseline_staleness(artifact, X, current)
    top = report["features"][0]
    assert top["feature"] == "f3"
    assert top["percentile_shift"] < -0.3
    f1 = next(r for r in report["features"] if r["feature"] == "f1")
    assert f1["current_missing_rate"] == pytest.approx(0.2)
    assert f1["reference_missing_rate"] == 0.0

    with pytest.raises(ValueError, match="feature columns"):
        baseline_staleness(artifact, X, X[:, :2])
