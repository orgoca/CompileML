"""Fairness audit: the exactness claims, and the refusals that protect them.

The headline property is that a group score gap decomposes across features
with a zero residual. Most of what follows exists to pin that down and to
make sure the module refuses rather than approximates when it cannot deliver
it.
"""

import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact
from compileml.bands import quantile_bands
from compileml.compile import train_whitebox
from compileml.fairness import (
    FairnessAudit,
    attribution_concentration,
    attribution_disparity,
    model_interaction_structure,
    wilson_interval,
)
from compileml.runtime import decide

RNG = np.random.default_rng(5)
N, P = 3000, 6
FEATURES = [f"f{i}" for i in range(P)]
REASONS = {
    n: {"code": n.upper(), "negative": f"{n} raised risk", "positive": f"{n} lowered risk"}
    for n in FEATURES
}


@pytest.fixture(scope="module")
def data():
    X = RNG.standard_normal((N, P))
    g = (RNG.random(N) < 0.45).astype(int)
    X[:, 2] += 0.8 * g  # the group difference enters through one feature
    teacher = 1 / (1 + np.exp(-(1.2 * X[:, 0] + 0.7 * X[:, 2] - 0.6 * X[:, 1])))
    y = (RNG.random(N) < teacher).astype(int)
    return X, y, g, teacher


def _artifact(X, y, teacher, *, max_depth=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = train_whitebox(X, teacher, n_estimators=25, max_depth=max_depth, random_state=0)
        latent = np.clip(model.predict(X), 0, 1)
        return build_artifact(
            model,
            FEATURES,
            np.median(X, axis=0),
            quantile_bands(latent, n_bands=8),
            calibration_latent=latent,
            calibration_y=y,
            reasons=REASONS,
        )


@pytest.fixture(scope="module")
def audit(data):
    X, y, g, teacher = data
    art = _artifact(X, y, teacher)
    dec = [decide(art, [float(v) for v in r], include_contributions=True) for r in X]
    a = FairnessAudit(dec, y, g, labels={0: "A", 1: "B"}, artifact=art, X=X)
    a.compute_all()
    return a


# ------------------------------------------------------- the headline claim
def test_group_gap_decomposes_with_zero_residual(audit):
    """§6: per-feature contributions sum to the group gap — exactly."""
    s6 = audit.section(6)
    assert s6["mean_gap_half_micro"] == pytest.approx(s6["sum_of_feature_gaps"], abs=1e-6)
    # The shares are of the gap, so they total 100% however they are split.
    assert sum(r["share_pct"] for r in s6["by_feature"]) == pytest.approx(100.0, abs=1e-6)


def test_the_planted_driver_is_identified(audit):
    """The feature carrying the group difference should dominate the gap."""
    top = audit.section(6)["by_feature"][0]
    assert top["feature"] == "f2"
    assert abs(top["share_pct"]) > 50


def test_decomposition_refuses_without_contributions(data):
    """The default payload has no per-feature impacts; say so, do not guess."""
    X, y, g, teacher = data
    art = _artifact(X, y, teacher)
    plain = [decide(art, [float(v) for v in r]) for r in X[:200]]
    with pytest.raises(ValueError, match="include_contributions"):
        attribution_disparity(plain, g[:200], FEATURES)


def test_decomposition_refuses_when_attribution_is_not_exact(data):
    """Depth > 2 leaves a residual, so the decomposition would not sum. Refuse."""
    X, y, g, teacher = data
    deep = _artifact(X, y, teacher, max_depth=3)
    assert deep["runtime"]["exact_attribution"] is False
    dec = [decide(deep, [float(v) for v in r], include_contributions=True) for r in X[:400]]
    with pytest.raises(ValueError, match="residual"):
        attribution_disparity(dec, g[:400], FEATURES)


# ------------------------------------------------------------ §8 behaviour
def test_concentration_varies_by_group_unlike_interaction_share(audit):
    """§8 must describe people, not the model.

    Its predecessor measured main-effect versus interaction points, which on
    a real depth-2 artifact is 100% interaction for everybody — a property of
    the compiled model that cannot differ between groups. Concentration does.
    """
    s8 = audit.section(8)
    values = [v["effective_drivers"] for v in s8["groups"].values()]
    assert all(1.0 <= v <= P for v in values)
    assert values[0] != values[1]  # it discriminates at all


def test_concentration_is_one_driver_when_only_one_moves():
    """A decision explained by a single feature has exactly one driver."""
    dec = [
        {
            "contributions": [
                {"index": 0, "impact_half_micro": 500},
                {"index": 1, "impact_half_micro": 0},
            ]
        },
        {
            "contributions": [
                {"index": 0, "impact_half_micro": 250},
                {"index": 1, "impact_half_micro": 250},
            ]
        },
    ]
    r = attribution_concentration(dec, np.array([0, 1]), ["a", "b"])
    assert r["groups"]["0"]["effective_drivers"] == pytest.approx(1.0)
    assert r["groups"]["1"]["effective_drivers"] == pytest.approx(2.0)


def test_a_group_with_no_movement_reports_nan_not_a_number():
    """Averaging an empty slice must not fabricate a value."""
    dec = [
        {"contributions": [{"index": 0, "impact_half_micro": 100}]},
        {"contributions": [{"index": 0, "impact_half_micro": 0}]},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no 'mean of empty slice'
        r = attribution_concentration(dec, np.array([0, 1]), ["a"])
    assert r["groups"]["1"]["n_with_movement"] == 0
    assert np.isnan(r["groups"]["1"]["effective_drivers"])


def test_model_structure_is_reported_as_model_level(audit):
    """The context that makes a per-group interaction share meaningless."""
    s = model_interaction_structure(audit.artifact)
    assert s["n_main_effects"] + s["n_interactions"] > 0
    assert isinstance(s["all_pairwise"], bool)


# ------------------------------------------------------------ §11 and §3
def test_counterfactual_reports_inapplicable_rather_than_skipping(audit):
    """The attribute is not a model input here — say so, positively."""
    s11 = audit.section(11)
    assert s11["applicable"] is False
    assert "does not apply" in s11["reason"]


def test_counterfactual_runs_when_the_attribute_is_an_input(data):
    X, y, g, teacher = data
    Xg = np.column_stack([X, g.astype(float)])
    names = [*FEATURES, "GROUP"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = train_whitebox(Xg, teacher, n_estimators=25, max_depth=2, random_state=0)
        latent = np.clip(model.predict(Xg), 0, 1)
        art = build_artifact(
            model,
            names,
            np.median(Xg, axis=0),
            quantile_bands(latent, n_bands=8),
            calibration_latent=latent,
            calibration_y=y,
        )
    dec = [decide(art, [float(v) for v in r]) for r in Xg[:500]]
    a = FairnessAudit(dec, y[:500], g[:500], artifact=art, X=Xg[:500], protected_feature="GROUP")
    s11 = a.section(11)
    assert s11["applicable"] is True
    assert 0.0 <= s11["band_flip_rate"] <= 1.0


def test_adverse_impact_ratio_and_intervals(audit):
    s3 = audit.section(3)
    assert 0.0 < s3["adverse_impact_ratio"] < 10.0
    for grp in s3["groups"].values():
        lo, hi = grp["rate_ci"]
        assert lo <= grp["rate"] <= hi


def test_wilson_interval_is_honest_on_small_groups():
    """Forty applicants must not produce a confident interval."""
    lo, hi = wilson_interval(1, 5)
    assert 0.0 <= lo < hi <= 1.0
    assert hi - lo > 0.4  # genuinely wide
    assert wilson_interval(0, 0) == pytest.approx((float("nan"), float("nan")), nan_ok=True)


# --------------------------------------------------------------- the object
def test_sections_that_lack_inputs_skip_with_a_reason(data):
    """No artifact means no §7/§8/§11 — and the report must say why."""
    X, y, g, teacher = data
    art = _artifact(X, y, teacher)
    dec = [decide(art, [float(v) for v in r]) for r in X[:300]]
    a = FairnessAudit(dec, y[:300], g[:300])
    a.compute_all()
    assert "representation" in a.results
    assert "attribution_disparity" in a.skipped
    assert a.skipped["attribution_disparity"]  # non-empty explanation
    with pytest.raises(RuntimeError, match="unavailable"):
        a.section(6)


def test_misaligned_inputs_are_rejected(data):
    X, y, g, teacher = data
    art = _artifact(X, y, teacher)
    dec = [decide(art, [float(v) for v in r]) for r in X[:100]]
    with pytest.raises(ValueError, match="same length"):
        FairnessAudit(dec, y[:99], g[:100])


def test_summary_states_it_is_not_a_verdict(audit):
    text = audit.summary()
    assert "not a verdict" in text
    assert "does not certify compliance" in text
