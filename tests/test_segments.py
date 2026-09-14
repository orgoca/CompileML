"""Retention by segment and across cutoff ranges (#12).

A portfolio retention figure is an average. These tests plant a segment that
pays for the compression and require the report to find it — in retention,
in decisions across its cutoff range, and in whether its range can even hold
a cutoff on the band ladder.
"""

import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact
from compileml.bands import monotone_quantile_bands
from compileml.compile import train_whitebox
from compileml.runtime import decide
from compileml.tune import retention_by_segment, sweep_whitebox

N, TRAIN = 10_000, 7_000
FEATURES = [f"f{i}" for i in range(5)]


@pytest.fixture(scope="module")
def data():
    """The thin-file segment carries an interaction the thick file does not."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((N, 5))
    segment = np.where(X[:, 4] > 0.8, "thin_file", "thick_file")
    logit = 1.2 * X[:, 0] - 0.8 * X[:, 1] - 1.5
    logit += np.where(segment == "thin_file", 1.4 * X[:, 2] * X[:, 3], 0.0)
    teacher = 1 / (1 + np.exp(-logit))
    y = (rng.random(N) < teacher).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = train_whitebox(X[:TRAIN], teacher[:TRAIN], n_estimators=40, random_state=0)
        latent = np.clip(model.predict(X[:TRAIN]), 0, 1)
        artifact = build_artifact(
            model,
            FEATURES,
            np.median(X[:TRAIN], axis=0),
            monotone_quantile_bands(latent, y[:TRAIN], n_bands=10),
            calibration_latent=latent,
            calibration_y=y[:TRAIN],
        )
    holdout = slice(TRAIN, N)
    return artifact, X[holdout], y[holdout], teacher[holdout], segment[holdout]


RANGES = {"thin_file": (0.10, 0.17), "thick_file": (0.02, 0.06)}


def test_the_segment_that_paid_for_compression_is_named(data):
    artifact, X, y, teacher, segment = data
    report = retention_by_segment(artifact, teacher, X, y, segments=segment, cutoff_ranges=RANGES)
    thin, thick = report["segments"]["thin_file"], report["segments"]["thick_file"]
    assert thin["gini_retention_pct"] < thick["gini_retention_pct"] - 10
    assert (
        thin["decisions"]["max_disagreement_rate"] > 2 * thick["decisions"]["max_disagreement_rate"]
    )
    assert report["worst_retention_segment"] == "thin_file"
    assert report["worst_disagreement_segment"] == "thin_file"


def test_an_artifact_judged_against_itself_disagrees_nowhere(data):
    """Same ranking, same volume: not one applicant decided differently."""
    artifact, X, y, _, _ = data
    own = np.array([decide(artifact, [float(v) for v in r], explain=False)["raw_micro"] for r in X])
    report = retention_by_segment(artifact, own, X, y, cutoff_ranges=(0.03, 0.30))
    entry = report["segments"]["all"]
    assert entry["gini_retention_pct"] == pytest.approx(100.0)
    assert all(p["disagreement_rate"] == 0.0 for p in entry["decisions"]["points"])
    assert entry["decisions"]["max_bad_rate_gap"] == 0.0


def test_agreement_compares_equal_volumes_across_the_whole_range(data):
    artifact, X, y, teacher, segment = data
    report = retention_by_segment(
        artifact, teacher, X, y, segments=segment, cutoff_ranges=RANGES, n_cutoffs=11
    )
    points = report["segments"]["thick_file"]["decisions"]["points"]
    assert len(points) == 11
    assert points[0]["cutoff_pd"] == pytest.approx(0.02)
    assert points[-1]["cutoff_pd"] == pytest.approx(0.06)
    rates = [p["approval_rate"] for p in points]
    assert rates == sorted(rates)  # a higher PD cutoff never approves fewer


def test_a_range_between_two_band_edges_cannot_hold_a_cutoff(data):
    artifact, X, y, teacher, _ = data
    wide = retention_by_segment(artifact, teacher, X, y, cutoff_ranges=(1e-6, 1 - 1e-6))
    listed = wide["segments"]["all"]["bands"]["edge_pds_in_range"]
    assert len(listed) == len(artifact["bands"]["edges_int"]) - 2  # every interior edge
    edge_pds = sorted(set(listed))
    gaps = [(a, b) for a, b in zip(edge_pds, edge_pds[1:]) if b - a > 1e-3]
    lo, hi = gaps[0]
    narrow = retention_by_segment(
        artifact, teacher, X, y, cutoff_ranges=(lo + (hi - lo) / 3, hi - (hi - lo) / 3)
    )
    bands = narrow["segments"]["all"]["bands"]
    assert bands["edges_in_range"] == 0
    assert bands["cutoff_expressible"] is False


def test_ranges_are_pd_fractions_and_name_real_segments(data):
    artifact, X, y, teacher, segment = data
    with pytest.raises(ValueError, match="PD fractions"):
        retention_by_segment(artifact, teacher, X, y, cutoff_ranges=(2, 6))
    with pytest.raises(ValueError, match="not present"):
        retention_by_segment(
            artifact, teacher, X, y, segments=segment, cutoff_ranges={"thin_fille": (0.1, 0.2)}
        )


def test_one_range_for_all_or_some_segments_without(data):
    artifact, X, y, teacher, segment = data
    shared = retention_by_segment(
        artifact, teacher, X, y, segments=segment, cutoff_ranges=(0.05, 0.1)
    )
    assert all(v["decisions"] for v in shared["segments"].values())

    partial = retention_by_segment(
        artifact, teacher, X, y, segments=segment, cutoff_ranges={"thin_file": (0.1, 0.17)}
    )
    assert partial["segments"]["thick_file"]["decisions"] is None
    assert partial["segments"]["thick_file"]["gini_retention_pct"] is not None


def test_a_single_class_segment_reports_nothing_rather_than_nonsense(data):
    artifact, X, y, teacher, _ = data
    labels = np.where(y == 1, "defaults_only", "no_defaults")
    report = retention_by_segment(artifact, teacher, X, y, segments=labels)
    only = report["segments"]["defaults_only"]
    assert np.isnan(only["gini"]) and only["gini_retention_pct"] is None


def test_sweep_names_the_worst_segment(data):
    artifact, X, y, teacher, segment = data
    rows = sweep_whitebox(
        X[:2000],
        teacher[:2000],
        y[:2000],
        trees_grid=(10,),
        depth_grid=(1, 2),
        X_val=X[2000:],
        y_val=y[2000:],
        teacher_latent_val=teacher[2000:],
        segments=segment[2000:],
        reference=0.5,
    )
    for row in rows:
        assert set(row["segments"]) == {"thin_file", "thick_file"}
        assert row["worst_segment"] == "thin_file"
    with pytest.raises(ValueError, match="evaluation row"):
        sweep_whitebox(
            X[:2000],
            teacher[:2000],
            y[:2000],
            trees_grid=(10,),
            depth_grid=(1,),
            segments=segment[:10],
            reference=0.5,
        )


# --------------------------------------------- acting on it: sample_weight
@pytest.fixture(scope="module")
def starved():
    """A 10% segment driven by its own features: a budget gap, not a structural one."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((12_000, 6))
    thin = X[:, 5] > 1.28
    logit = np.where(thin, 1.3 * X[:, 2] - 1.1 * X[:, 3] - 1.0, 1.2 * X[:, 0] - 0.8 * X[:, 1] - 2.0)
    teacher = 1 / (1 + np.exp(-logit))
    y = (rng.random(len(X)) < teacher).astype(int)
    return X, y, teacher, np.where(thin, "thin_file", "thick_file")


def test_weighting_moves_the_budget_to_a_starved_segment_and_others_pay(starved):
    X, y, teacher, segment = starved
    tr, te = slice(0, 8000), slice(8000, None)

    def sweep(weights):
        (row,) = sweep_whitebox(
            X[tr],
            teacher[tr],
            y[tr],
            trees_grid=(15,),
            depth_grid=(2,),
            X_val=X[te],
            y_val=y[te],
            teacher_latent_val=teacher[te],
            segments=segment[te],
            sample_weight=weights,
            reference=0.5,
        )
        return row["segments"]

    plain = sweep(None)
    weighted = sweep(np.where(segment[tr] == "thin_file", 5.0, 1.0))
    assert (
        weighted["thin_file"]["gini_retention_pct"] > plain["thin_file"]["gini_retention_pct"] + 30
    )
    # The budget moved; it was not created. The other segment pays.
    assert weighted["thick_file"]["gini_retention_pct"] < plain["thick_file"]["gini_retention_pct"]


def test_a_weighted_whitebox_is_still_exact(starved):
    X, y, teacher, segment = starved
    weights = np.where(segment == "thin_file", 5.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = train_whitebox(
            X, teacher, n_estimators=15, random_state=0, sample_weight=weights
        )
        latent = np.clip(model.predict(X), 0, 1)
        artifact = build_artifact(
            model,
            [f"f{i}" for i in range(6)],
            np.median(X, axis=0),
            monotone_quantile_bands(latent, y, n_bands=8),
            calibration_latent=latent,
            calibration_y=y,
        )
    assert artifact["runtime"]["exact_attribution"] is True
    for row in X[:50]:
        assert decide(artifact, [float(v) for v in row])["attribution_residual_half_micro"] == 0


def test_sample_weight_is_validated(starved):
    X, _, teacher, _ = starved
    with pytest.raises(ValueError, match="one weight per row"):
        train_whitebox(X, teacher, sample_weight=np.ones(10))
    with pytest.raises(ValueError, match="non-negative"):
        train_whitebox(X, teacher, sample_weight=-np.ones(len(X)))


def test_cutoff_ranges_refuse_an_uncalibrated_artifact(data):
    import copy

    artifact, X, y, teacher, segment = data
    raw = copy.deepcopy(artifact)
    raw["calibration"] = None
    with pytest.raises(ValueError, match="no calibration table"):
        retention_by_segment(raw, teacher, X, y, cutoff_ranges=(0.05, 0.1))
    retention_by_segment(raw, teacher, X, y, segments=segment)  # retention alone is fine
