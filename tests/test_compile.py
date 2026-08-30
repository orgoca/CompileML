"""Compile-side tests: distill → extract → quantize → build → decide round trips."""

import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact, fit_isotonic_table, save_artifact
from compileml.compile import (
    extract_trees,
    max_depth,
    quantization_error_bound,
    quantize_model,
    score_float,
    train_whitebox,
    validate_extraction,
)
from compileml.runtime import decide, load_artifact, score_micro, verify_artifact
from compileml.runtime.calibrate import calibrate_ppm

RNG = np.random.default_rng(7)
N, P = 4000, 8
FEATURES = [f"f{i}" for i in range(P)]


@pytest.fixture(scope="module")
def data():
    X = RNG.standard_normal((N, P))
    teacher = 1.0 / (1.0 + np.exp(-(1.4 * X[:, 0] + X[:, 1] * X[:, 2] + 0.6 * X[:, 3])))
    y = (RNG.random(N) < teacher).astype(int)
    return X, teacher, y


@pytest.fixture(scope="module")
def whitebox(data):
    X, teacher, _ = data
    model, metrics = train_whitebox(X, teacher, n_estimators=120, random_state=0)
    assert metrics["spearman"] > 0.9
    return model


@pytest.fixture(scope="module")
def artifact(data, whitebox):
    X, _, y = data
    latent = np.clip(whitebox.predict(X), 0.0, 1.0)
    edges = np.quantile(latent, np.linspace(0, 1, 9))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # reason coverage warning tested separately
        return build_artifact(
            whitebox,
            FEATURES,
            np.median(X, axis=0),
            edges,
            calibration_latent=latent,
            calibration_y=y,
            X_sample=X[:300],
        )


# ------------------------------------------------------------- quantization
def test_extract_matches_sklearn_predict(data, whitebox):
    X, _, _ = data
    extracted = extract_trees(whitebox)
    for row in X[:50]:
        assert score_float(extracted, [float(v) for v in row]) == pytest.approx(
            float(whitebox.predict(row.reshape(1, -1))[0]), abs=1e-9
        )


def test_constant_init_folded_into_base(data, whitebox):
    """Default init is a DummyRegressor constant; it belongs in base."""
    extracted = extract_trees(whitebox)
    assert extracted.base == pytest.approx(float(whitebox.init_.constant_.ravel()[0]))


def test_init_zero_extracts_zero_base(data):
    """sklearn stores init='zero' as the literal string — base 0.0 is correct."""
    from sklearn.ensemble import GradientBoostingRegressor

    X, teacher, _ = data
    model = GradientBoostingRegressor(n_estimators=5, max_depth=2, init="zero")
    model.fit(X, teacher)
    extracted = extract_trees(model)  # parity-gated inside
    assert extracted.base == 0.0
    for row in X[:20]:
        assert score_float(extracted, [float(v) for v in row]) == pytest.approx(
            float(model.predict(row.reshape(1, -1))[0]), abs=1e-9
        )


def test_classifier_extraction_raises(data):
    """A classifier's log-odds prior lives nowhere in the artifact.

    Before this was caught, ``base`` silently became 0.0 and every compiled
    margin was short by the prior — a constant offset, so Gini, Spearman and
    quantile edges all agreed while calibrated PD and the band ladder did not.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    X, _, y = data
    model = GradientBoostingClassifier(n_estimators=5, max_depth=2)
    model.fit(X, y)
    with pytest.raises(ValueError, match="log-odds"):
        extract_trees(model)


def test_estimator_init_raises(data):
    """A non-constant init makes the trees a residual with no base to add."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    X, teacher, _ = data
    model = GradientBoostingRegressor(n_estimators=5, max_depth=2, init=LinearRegression())
    model.fit(X, teacher)
    with pytest.raises(ValueError, match="not a constant"):
        extract_trees(model)


def test_sklearn_extraction_is_parity_gated(data, whitebox):
    """The sklearn family is no longer exempt from validate_extraction."""
    extracted = extract_trees(whitebox)
    extracted.base += 0.01  # the exact shape the classifier bug took
    with pytest.raises(ValueError, match="extraction parity failed for sklearn"):
        validate_extraction(extracted, whitebox)


def test_quantization_error_within_bound(data, whitebox):
    X, _, _ = data
    extracted = extract_trees(whitebox)
    model_int = quantize_model(extracted)
    bound = quantization_error_bound(model_int)
    for row in X[:200]:
        row_f = [float(v) for v in row]
        err = abs(score_float(extracted, row_f) - score_micro(model_int, row_f) / 1_000_000)
        assert err <= bound


def test_measured_depth(whitebox):
    model_int = quantize_model(extract_trees(whitebox))
    assert max_depth(model_int) == 2


# ---------------------------------------------------------------- artifact
def test_artifact_verifies_and_decides(data, artifact):
    X, _, _ = data
    assert verify_artifact(artifact)
    assert artifact["runtime"]["exact_attribution"] is True
    assert artifact["metadata"]["model_family"] == "sklearn"
    out = decide(artifact, [float(v) for v in X[0]])
    assert out["attribution_residual_half_micro"] == 0
    assert 0 <= out["latent_int"] <= 1000
    assert 0.0 <= out["pd"] <= 1.0


def test_artifact_rank_fidelity(data, whitebox, artifact):
    from scipy.stats import spearmanr

    X, _, _ = data
    float_latent = np.clip(whitebox.predict(X[:500]), 0, 1)
    int_latent = [
        decide(artifact, [float(v) for v in row], explain=False)["latent_micro"] for row in X[:500]
    ]
    rho = spearmanr(float_latent, int_latent)[0]
    assert rho > 0.9999


def test_save_load_roundtrip(tmp_path, data, artifact):
    X, _, _ = data
    path = tmp_path / "model.json"
    save_artifact(artifact, path)
    loaded = load_artifact(path)  # verifies hash
    a = decide(loaded, [float(v) for v in X[1]])
    b = decide(artifact, [float(v) for v in X[1]])
    for key in ("band", "latent_int", "pd_ppm", "raw_micro", "reasons_negative"):
        assert a[key] == b[key]


def test_reason_coverage_warns_and_records(data, whitebox):
    X, _, y = data
    latent = np.clip(whitebox.predict(X), 0, 1)
    edges = np.quantile(latent, np.linspace(0, 1, 5))
    partial_reasons = {"f0": {"code": "F0", "negative": "n", "positive": "p"}}
    with pytest.warns(UserWarning, match="reason dictionary covers 1/8"):
        art = build_artifact(
            whitebox, FEATURES, np.median(X, axis=0), edges, reasons=partial_reasons
        )
    assert art["metadata"]["reason_coverage"] == pytest.approx(1 / 8)


def test_band_edge_collision_raises(data, whitebox):
    X, _, _ = data
    edges = [0.0, 0.10001, 0.10002, 1.0]  # collide at scale 1000
    with pytest.raises(ValueError, match="collide"):
        build_artifact(whitebox, FEATURES, np.median(X, axis=0), edges)


def test_depth3_warns_and_reports_inexact(data):
    X, teacher, _ = data
    with pytest.warns(UserWarning, match="max_depth=3"):
        deep, _ = train_whitebox(X, teacher, n_estimators=20, max_depth=3, random_state=0)
    latent = np.clip(deep.predict(X), 0, 1)
    edges = np.quantile(latent, np.linspace(0, 1, 5))
    with pytest.warns(UserWarning, match="depth is 3"):
        art = build_artifact(deep, FEATURES, np.median(X, axis=0), edges)
    assert art["runtime"]["exact_attribution"] is False
    out = decide(art, [float(v) for v in X[0]])
    # Identity still holds exactly — the residual absorbs >2-way interactions.
    lhs = 2 * (out["raw_micro"] - out["baseline_micro"])
    assert lhs == out["attribution_sum_half_micro"] + out["attribution_residual_half_micro"]


# -------------------------------------------------------------- calibration
def test_calibration_table_close_to_sklearn(data, whitebox):
    from sklearn.isotonic import IsotonicRegression

    X, _, y = data
    latent = np.clip(whitebox.predict(X), 0, 1)
    table = fit_isotonic_table(latent, y)
    assert all(b > a for a, b in zip(table["f_micro"], table["f_micro"][1:]))
    assert all(b >= a for a, b in zip(table["pd_ppm"], table["pd_ppm"][1:]))

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(latent, y)
    probe = np.linspace(0.02, 0.98, 25)
    for v in probe:
        expected = float(iso.predict([v])[0])
        got = calibrate_ppm(int(round(v * 1_000_000)), table, 1_000_000) / 1_000_000
        assert got == pytest.approx(expected, abs=1e-3)


# ------------------------------------------------------- other model familes
def test_xgboost_direct_compile():
    xgb = pytest.importorskip("xgboost")
    X = RNG.standard_normal((2000, 5)).astype(np.float32)
    target = 1.0 / (1.0 + np.exp(-(X[:, 0] + 0.5 * X[:, 1] * X[:, 2])))
    model = xgb.XGBRegressor(n_estimators=25, max_depth=2, learning_rate=0.3, random_state=0)
    model.fit(X, target)

    extracted = extract_trees(model)  # includes parity validation vs booster
    assert extracted.family == "xgboost"
    assert extracted.input_precision == "float32"

    latent = np.clip(model.predict(X), 0, 1)
    edges = np.quantile(latent, np.linspace(0, 1, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        art = build_artifact(
            model, [f"f{i}" for i in range(5)], np.median(X, axis=0), edges, X_sample=X[:200]
        )
    assert art["model"]["input_precision"] == "float32"
    # Integer artifact vs float booster: within quantization bound everywhere.
    bound = art["metadata"]["quantization"]["error_bound"] + 1e-6
    for row in X[:200]:
        int_latent = decide(art, [float(v) for v in row], explain=False)["raw_micro"] / 1e6
        assert abs(int_latent - float(model.predict(row.reshape(1, -1))[0])) <= bound


def test_lightgbm_direct_compile():
    lgb = pytest.importorskip("lightgbm")
    X = RNG.standard_normal((2000, 5))
    target = 1.0 / (1.0 + np.exp(-(X[:, 0] - 0.7 * X[:, 3])))
    model = lgb.LGBMRegressor(n_estimators=25, max_depth=2, learning_rate=0.3, verbose=-1)
    model.fit(X, target)

    extracted = extract_trees(model)  # includes parity validation vs booster
    assert extracted.family == "lightgbm"
    assert extracted.input_precision == "float64"


def test_unsupported_model_raises():
    with pytest.raises(TypeError, match="not compilable"):
        extract_trees(object())


def test_threshold_decimals_quantization(data, whitebox):
    X, _, y = data
    latent = np.clip(whitebox.predict(X), 0, 1)
    edges = np.quantile(latent, np.linspace(0, 1, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        art = build_artifact(
            whitebox,
            FEATURES,
            np.median(X, axis=0),
            edges,
            threshold_decimals=4,
            X_sample=X[:100],
        )
    for tree in art["model"]["trees"]:
        for t in tree["threshold"]:
            assert t == round(t, 4)
    assert art["metadata"]["quantization"]["threshold_decimals"] == 4
    out = decide(art, [float(v) for v in X[0]], explain=False)
    assert 0 <= out["latent_int"] <= 1000
    # Validation's fidelity check mirrors the quantization and still passes.
    from compileml.validate import validate_artifact

    report = validate_artifact(art, X_val=X[:300], model=whitebox)
    assert report["checks"]["3_fidelity"]["pass"]
