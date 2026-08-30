"""The reference floor, the target axis, and validation check 10.

(Not to be confused with ``test_reference.py``, which guards the committed
determinism oracle — a different sense of the word.)

The point of all three: ``gini_retention_pct`` is one-sided. It reports
distance to a teacher ceiling and is structurally incapable of saying that a
handful of logistic coefficients scored higher. These tests pin the other
side of the comparison, and the knob that lets the training target be
measured rather than assumed.
"""

import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact
from compileml.compile import train_whitebox
from compileml.reference import ReferenceModel, fit_reference, reference_gini
from compileml.tune import sweep_whitebox
from compileml.validate import validate_artifact

RNG = np.random.default_rng(23)
N, P = 4000, 5
FEATURES = [f"x{i}" for i in range(P)]


@pytest.fixture(scope="module")
def data():
    X = RNG.standard_normal((N, P))
    teacher = 1.0 / (1.0 + np.exp(-(1.4 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2])))
    y = (RNG.random(N) < teacher).astype(int)
    return X, teacher, y


@pytest.fixture(scope="module")
def reference(data):
    X, _, y = data
    return fit_reference(X[:3000], y[:3000], feature_names=FEATURES)


def _artifact(data, n_estimators, max_depth=2, n_bands=8):
    X, teacher, y = data
    model, _ = train_whitebox(
        X[:3000], teacher[:3000], n_estimators=n_estimators, max_depth=max_depth, random_state=1
    )
    latent = np.clip(model.predict(X[:3000]), 0.0, 1.0)
    edges = np.unique(np.quantile(latent, np.linspace(0, 1, n_bands + 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_artifact(
            model,
            FEATURES,
            np.median(X, axis=0),
            edges,
            calibration_latent=latent,
            calibration_y=y[:3000],
        )


# ------------------------------------------------------------- the reference
def test_fit_reference_shape_and_evidence(reference):
    assert isinstance(reference, ReferenceModel)
    assert reference.kind == "woe_logit"
    assert reference.n_coefficients == P
    assert set(reference.information_value) == set(FEATURES)
    # The generating features carry the information; the noise ones do not.
    iv = reference.information_value
    assert iv["x0"] > iv["x3"] and iv["x0"] > iv["x4"]
    assert all(v >= 0 for v in iv.values())  # IV is non-negative by construction
    # One WOE value per bin, one fewer threshold than bins.
    for thresholds, woe in zip(reference.thresholds, reference.woe):
        assert len(woe) == len(thresholds) + 1


def test_reference_gini_on_holdout(reference, data):
    X, _, y = data
    assert 0.3 < reference_gini(reference, X[3000:], y[3000:]) < 1.0


def test_reference_gini_accepts_a_bare_float():
    """A champion scorecard's Gini is a better floor than any fitted here."""
    assert reference_gini(0.8515, None, None) == pytest.approx(0.8515)


def test_reference_tolerates_missing_values(reference, data):
    X, _, _ = data
    X_nan = X[3000:].copy()
    X_nan[0, 0] = np.nan
    assert np.isfinite(reference.score(X_nan)).all()


def test_fit_reference_rejects_bad_input(data):
    X, _, y = data
    with pytest.raises(ValueError, match="unknown reference kind"):
        fit_reference(X[:500], y[:500], kind="random_forest")
    with pytest.raises(ValueError, match="feature_names"):
        fit_reference(X[:500], y[:500], feature_names=["a", "b"])
    with pytest.raises(ValueError, match="both classes"):
        fit_reference(X[:500], np.zeros(500, dtype=int))


# ------------------------------------------------------ the target axis (4.2)
def test_alpha_axis_sweeps_the_target(data, reference):
    X, teacher, y = data
    rows = sweep_whitebox(
        X,
        teacher,
        y,
        trees_grid=(20,),
        depth_grid=(2,),
        alpha_grid=(0.0, 0.5, 1.0),
        reference=reference,
        random_state=0,
    )
    assert len(rows) == 3
    assert [r["alpha"] for r in rows] == [0.0, 0.5, 1.0]
    # Distinct targets must produce distinct models, or the axis does nothing.
    assert len({r["gini"] for r in rows}) > 1


def test_alpha_defaults_to_distillation(data):
    """alpha_grid defaults to (0.0,) — the historical behavior, unchanged."""
    X, teacher, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = sweep_whitebox(X, teacher, y, trees_grid=(10,), depth_grid=(1, 2), random_state=0)
    assert len(rows) == 2
    assert all(r["alpha"] == 0.0 for r in rows)


def test_sweep_without_a_teacher(data, reference):
    """No teacher: alpha is forced to labels and teacher columns go None."""
    X, _, y = data
    rows = sweep_whitebox(
        X, None, y, trees_grid=(10,), depth_grid=(2,), reference=reference, random_state=0
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["alpha"] == 1.0
    assert row["gini_retention_pct"] is None
    assert row["spearman_vs_teacher"] is None
    assert row["gini_vs_reference_pct"] is not None


def test_train_whitebox_teacher_latent_alias(data):
    X, teacher, _ = data
    with pytest.warns(DeprecationWarning, match="teacher_latent"):
        model, _ = train_whitebox(X[:500], teacher_latent=teacher[:500], n_estimators=3)
    assert model is not None
    with pytest.raises(TypeError, match="not both"):
        train_whitebox(X[:500], teacher[:500], teacher_latent=teacher[:500], n_estimators=3)
    with pytest.raises(TypeError, match="missing required argument"):
        train_whitebox(X[:500], n_estimators=3)


# --------------------------------------------------- the floor in the sweep
def test_sweep_reports_the_floor_beside_the_ceiling(data, reference):
    X, teacher, y = data
    rows = sweep_whitebox(
        X, teacher, y, trees_grid=(20,), depth_grid=(2,), reference=reference, random_state=0
    )
    row = rows[0]
    assert row["reference_gini"] is not None
    assert row["gini_vs_reference_pct"] == pytest.approx(
        100 * row["gini"] / row["reference_gini"], abs=0.05
    )
    assert isinstance(row["beats_reference"], bool)


def test_sweep_warns_on_one_sided_retention(data):
    """Teacher retention without a floor is the reporting flaw itself."""
    X, teacher, y = data
    with pytest.warns(UserWarning, match="without a reference"):
        sweep_whitebox(X, teacher, y, trees_grid=(10,), depth_grid=(2,), random_state=0)


# ---------------------------------------------------------------- check 10
def test_check10_passes_a_healthy_artifact(data, reference):
    X, _, y = data
    result = validate_artifact(
        _artifact(data, 80),
        X_val=X[3000:],
        y_val=y[3000:],
        reference=reference,
        require_reference_floor=True,
    )
    check = result["checks"]["10_reference_floor"]
    assert check["pass"] and check["clears_floor"] and not check["skipped"]


def test_check10_fails_a_starved_artifact(data, reference):
    """5 trees at depth 1 should lose to a logistic regression, and be told so."""
    X, _, y = data
    starved = _artifact(data, 5, max_depth=1, n_bands=3)
    result = validate_artifact(
        starved,
        X_val=X[3000:],
        y_val=y[3000:],
        reference=reference,
        require_reference_floor=True,
    )
    check = result["checks"]["10_reference_floor"]
    assert not check["pass"] and not check["clears_floor"]
    assert not result["all_pass"]
    assert check["artifact_gini"] < check["reference_gini"]

    # Advisory by default: evidence recorded, gate not applied.
    advisory = validate_artifact(starved, X_val=X[3000:], y_val=y[3000:], reference=reference)
    assert advisory["checks"]["10_reference_floor"]["pass"]
    assert not advisory["checks"]["10_reference_floor"]["clears_floor"]


def test_check10_skips_without_a_reference(data):
    X, _, y = data
    result = validate_artifact(_artifact(data, 40), X_val=X[3000:], y_val=y[3000:])
    assert result["checks"]["10_reference_floor"]["skipped"]
    assert result["checks"]["10_reference_floor"]["pass"]
