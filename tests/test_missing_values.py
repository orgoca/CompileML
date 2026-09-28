"""Missing values are refused, not mis-compiled.

The artifact has no missing-value branch: it imputes the baseline. A model
that learned where NaN goes would score those rows one way and the artifact
another, so every path that could produce that pairing refuses it.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from compileml.artifact import build_artifact
from compileml.bands import monotone_quantile_bands
from compileml.compile import extract_trees, train_whitebox
from compileml.select import compile_selected

N, P = 3000, 4
NAMES = [f"f{i}" for i in range(P)]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((N, P))
    logit = X[:, 0] - 0.6 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3]
    y = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
    X_nan = X.copy()
    X_nan[rng.random(N) < 0.2, 0] = np.nan
    X_nan[rng.random(N) < 0.05, 3] = np.nan
    return X, X_nan, y


def _build(model, X, y, baseline=None):
    latent = np.clip(model.predict(X), 0.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_artifact(
            model,
            NAMES,
            np.median(X, axis=0) if baseline is None else baseline,
            monotone_quantile_bands(latent, y, n_bands=5),
            calibration_latent=latent,
            calibration_y=y,
        )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"backend": "gbr"}, {"backend": "hist"}, {"monotone_constraints": [1, 0, 0, 0]}],
)
def test_train_whitebox_refuses_nan_on_every_backend(data, kwargs):
    _, X_nan, y = data
    with pytest.raises(ValueError, match=r"missing values \(NaN\) in 2 column\(s\): \[0, 3\]"):
        train_whitebox(X_nan, y.astype(float), n_estimators=5, **kwargs)


def test_compile_selected_refuses_nan_before_the_ceiling_runs(data):
    _, X_nan, y = data
    calls = []

    def ceiling(Xf, yf):
        calls.append(1)
        raise AssertionError("the ceiling must not be fitted")

    with pytest.raises(ValueError, match=r"missing values \(NaN\).*\['f0', 'f3'\]"):
        compile_selected(X_nan, y, ceiling=ceiling, feature_names=NAMES)
    assert calls == []


def test_extraction_refuses_a_model_trained_on_nan(data):
    _, X_nan, y = data
    model = HistGradientBoostingRegressor(
        max_iter=30, max_depth=2, early_stopping=False, random_state=0
    ).fit(X_nan, y.astype(float))
    with pytest.raises(ValueError, match=r"trained on missing values.*\[0"):
        extract_trees(model)


@pytest.mark.parametrize("seed", range(4))
def test_extraction_accepts_models_that_never_saw_nan(data, seed):
    """HGB records a missing direction on every split; without NaN it is the default."""
    X, _, y = data
    weights = np.random.default_rng(seed).random(N) + 0.5 if seed % 2 else None
    common = {"max_iter": 40, "max_depth": 2, "early_stopping": False, "random_state": seed}
    for model in (
        HistGradientBoostingRegressor(**common),
        HistGradientBoostingRegressor(**common, monotonic_cst=[1, -1, 0, 0]),
    ):
        model.fit(X, y.astype(float), sample_weight=weights)
        assert extract_trees(model).trees


def test_build_refuses_a_non_finite_baseline(data):
    X, _, y = data
    model, _ = train_whitebox(X, y.astype(float), n_estimators=10, backend="hist")
    for bad in (np.nan, np.inf):
        baseline = np.median(X, axis=0)
        baseline[2] = bad
        with pytest.raises(ValueError, match=r"baseline is not finite for \['f2'\]"):
            _build(model, X, y, baseline=baseline)
    art = _build(model, X, y)
    assert all(np.isfinite(art["features"]["baseline"]))


def test_build_refuses_a_non_finite_threshold(data):
    X, _, y = data
    model, _ = train_whitebox(X, y.astype(float), n_estimators=10, backend="hist")
    extracted = extract_trees(model)
    tree = dict(extracted.trees[0])
    split = next(i for i, f in enumerate(tree["feature"]) if f >= 0)
    tree["threshold"] = list(tree["threshold"])
    tree["threshold"][split] = float("inf")
    tampered = replace(extracted, trees=[tree, *extracted.trees[1:]])
    latent = np.clip(model.predict(X), 0.0, 1.0)
    with pytest.raises(ValueError, match="non-finite threshold"):
        build_artifact(
            tampered, NAMES, np.median(X, axis=0), monotone_quantile_bands(latent, y, n_bands=5)
        )
