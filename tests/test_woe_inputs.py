"""WOE-transformed inputs: what docs/howto/woe.md says about them.

A weight-of-evidence column is an ordinary numeric column that takes a handful
of values, one per bin. These tests pin the statements that page makes, on
synthetic data with a fixed seed: four raw attributes cut into 3-4 bins each
and replaced by the bin's WOE (higher WOE = lower risk).
"""

import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact
from compileml.bands import governance_bands
from compileml.compile import extract_trees, train_whitebox
from compileml.runtime import decide

NAMES = ["woe_util", "woe_late", "woe_tenure", "woe_income"]
N = 6000


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    util = rng.beta(2, 3, N)
    late = rng.poisson(0.6, N)
    tenure = rng.gamma(3, 2, N)
    income = rng.lognormal(0, 0.5, N)
    logit = -1.4 + 2.0 * util + 0.5 * np.minimum(late, 3) - 0.08 * tenure - 0.3 * np.log(income)
    y = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
    bins = np.column_stack(
        [
            np.digitize(util, [0.2, 0.4, 0.6]),
            np.digitize(late, [1, 2]),
            np.digitize(tenure, [3, 6, 10]),
            np.digitize(income, [0.7, 1.0, 1.5]),
        ]
    )
    woe = np.zeros(bins.shape)
    for j in range(bins.shape[1]):
        for b in np.unique(bins[:, j]):
            m = bins[:, j] == b
            bad, good = y[m].sum() + 0.5, (1 - y[m]).sum() + 0.5
            woe[m, j] = np.log((good / (y == 0).sum()) / (bad / (y == 1).sum()))
    return woe, y


def _modal(X):
    """The most common WOE value of each column: a bin, not an average of bins."""
    return np.array(
        [
            np.unique(c, return_counts=True)[0][np.argmax(np.unique(c, return_counts=True)[1])]
            for c in X.T
        ]
    )


def _build(model, X, y, **kw):
    latent = np.clip(model.predict(X), 0.0, 1.0)
    kw.setdefault("baseline", _modal(X))
    baseline = kw.pop("baseline")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # reason coverage warning tested elsewhere
        return build_artifact(
            model,
            NAMES,
            baseline,
            np.quantile(latent, np.linspace(0, 1, 6)),
            calibration_latent=latent,
            calibration_y=y,
            X_sample=X[:500],
            **kw,
        )


@pytest.fixture(scope="module")
def model(data):
    X, y = data
    return train_whitebox(X, y.astype(float), n_estimators=40)[0]


@pytest.fixture(scope="module")
def artifact(data, model):
    X, y = data
    return _build(model, X, y)


def test_woe_features_compile_and_match_the_source_model(data, model, artifact):
    X, _ = data
    assert all(len(np.unique(X[:, j])) <= 4 for j in range(X.shape[1]))
    bound = artifact["metadata"]["quantization"]["error_bound"] + 1e-6
    for row in X[:300]:
        got = decide(artifact, [float(v) for v in row], explain=False)["raw_micro"] / 1e6
        assert abs(got - float(model.predict(row.reshape(1, -1))[0])) <= bound


def test_split_thresholds_fall_between_woe_values(data, model):
    """A split can only separate bins, so thresholds sit between observed values."""
    X, _ = data
    extracted = extract_trees(model)
    for tree in extracted.trees:
        for f, thr in zip(tree["feature"], tree["threshold"]):
            if f < 0:
                continue
            values = np.unique(X[:, f])
            assert thr not in values
            assert values.min() < thr < values.max()


@pytest.mark.parametrize("family", ["xgboost", "lightgbm"])
def test_boosting_libraries_compile_woe_features(data, family):
    lib = pytest.importorskip(family)
    X, y = data
    if family == "xgboost":
        model = lib.XGBRegressor(n_estimators=20, max_depth=2, learning_rate=0.3, random_state=0)
    else:
        model = lib.LGBMRegressor(n_estimators=20, max_depth=2, learning_rate=0.3, verbose=-1)
    model.fit(X, y.astype(float))
    art = _build(model, X, y)
    bound = art["metadata"]["quantization"]["error_bound"] + 1e-6
    for row in X[:300]:
        got = decide(art, [float(v) for v in row], explain=False)["raw_micro"] / 1e6
        assert abs(got - float(model.predict(row.reshape(1, -1))[0])) <= bound


def test_rows_with_the_same_woe_values_share_a_band(data, artifact):
    """Coarse bins make the latent a plateau, so edges cannot split a bin combination."""
    X, _ = data
    seen: dict[tuple, int] = {}
    for row in X[:2000]:
        out = decide(artifact, [float(v) for v in row], explain=False)
        assert seen.setdefault(tuple(row), out["band_idx"]) == out["band_idx"]
    assert len(seen) < 2000


def test_governance_bands_cap_within_band_auc_on_woe_latent(data, model):
    X, y = data
    latent = np.clip(model.predict(X), 0.0, 1.0)
    spec = governance_bands(latent, y, min_band_size=300)
    assert spec.n_bands >= 2
    assert max(spec.metadata["band_auc"]) <= 0.60


REASONS = {
    "woe_util": {
        "code": "HIGH_UTILIZATION",
        "negative": "Credit utilization is high relative to available limits.",
        "positive": "Credit utilization is well managed.",
    },
    "woe_late": {
        "code": "LATE_PAYMENTS",
        "negative": "Payments were made after their due date.",
        "positive": "Payments were made on time.",
    },
    "woe_tenure": {
        "code": "SHORT_HISTORY",
        "negative": "The credit history is short.",
        "positive": "The credit history is long.",
    },
    "woe_income": {
        "code": "LOW_INCOME",
        "negative": "Reported income is low.",
        "positive": "Reported income is high.",
    },
}
LABELS = {
    "woe_util": "Credit utilization",
    "woe_late": "Late payments",
    "woe_tenure": "Length of credit history",
    "woe_income": "Income",
}


def test_reasons_carry_the_dictionary_text_for_the_underlying_attribute(data, model):
    X, y = data
    art = _build(model, X, y, reasons=REASONS, display_names=LABELS)
    base = _modal(X)
    util = np.unique(X[:, 0])

    worst = base.copy()
    worst[0] = util.min()  # lowest WOE = highest risk
    adverse = decide(art, list(worst))["reasons_negative"]
    assert adverse[0]["feature"] == "woe_util"
    assert adverse[0]["code"] == "HIGH_UTILIZATION"
    assert adverse[0]["label"] == "Credit utilization"
    assert adverse[0]["message"] == REASONS["woe_util"]["negative"]

    best = base.copy()
    best[0] = util.max()
    favorable = decide(art, list(best))["reasons_positive"]
    assert favorable[0]["code"] == "HIGH_UTILIZATION"
    assert favorable[0]["message"] == REASONS["woe_util"]["positive"]


def test_the_baseline_is_a_woe_value_and_a_row_at_the_baseline_has_no_reasons(
    data, model, artifact
):
    """Reasons are measured against the baseline row, not against a bin."""
    X, _ = data
    base = [float(v) for v in artifact["features"]["baseline"]]
    assert all(v in np.unique(X[:, j]) for j, v in enumerate(base))
    out = decide(artifact, base)
    assert out["reasons_negative"] == [] and out["reasons_positive"] == []
    assert out["attribution_sum_half_micro"] == 0


def test_declared_direction_is_in_woe_units(data):
    """Higher WOE is lower risk, so a risk-scored model is decreasing in WOE."""
    X, y = data
    cst = [-1, -1, -1, -1]
    model = train_whitebox(X, y.astype(float), n_estimators=40, monotone_constraints=cst)[0]
    art = _build(model, X, y, monotone_constraints=cst)
    assert art["model"]["monotone_constraints"] == cst

    base = [float(v) for v in art["features"]["baseline"]]
    for j in range(X.shape[1]):
        scores = []
        for v in np.unique(X[:, j]):
            row = list(base)
            row[j] = float(v)
            scores.append(decide(art, row, explain=False)["raw_micro"])
        assert scores == sorted(scores, reverse=True)

    with pytest.raises(ValueError, match="monotone constraint violated"):
        _build(model, X, y, monotone_constraints=[+1, -1, -1, -1])


def test_a_missing_bin_is_an_ordinary_woe_value_and_nan_is_not_that_bin(data):
    X, y = data
    X = X.copy()
    # Draw the "missing" rows from the riskier (bad) rows, so the trees have a reason
    # to split them off the baseline bin instead of leaving it to chance.
    gone = (y == 1) & (np.random.default_rng(3).random(N) < 0.3)
    X[gone, 3] = 0.9  # the WOE assigned to "income missing", scored like any bin
    model = train_whitebox(X, y.astype(float), n_estimators=40)[0]
    art = _build(model, X, y)
    base = [float(v) for v in art["features"]["baseline"]]

    in_missing_bin = list(base)
    in_missing_bin[3] = 0.9
    at_baseline = decide(art, base, explain=False)
    # A NaN reaching the artifact is replaced by the baseline, not by the missing bin.
    for j in range(len(base)):
        with_nan = list(base)
        with_nan[j] = float("nan")
        assert decide(art, with_nan, explain=False)["raw_micro"] == at_baseline["raw_micro"]
    assert decide(art, in_missing_bin, explain=False)["raw_micro"] != at_baseline["raw_micro"]

    X_nan = X.copy()
    X_nan[0, 3] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        train_whitebox(X_nan, y.astype(float))


def test_woe_table_version_can_travel_in_hashed_metadata(data, model):
    """The artifact holds split thresholds in WOE units, not the WOE table."""
    X, y = data
    a = _build(model, X, y, metadata={"woe_table": "v1"})
    b = _build(model, X, y, metadata={"woe_table": "v2"})
    assert a["metadata"]["woe_table"] == "v1"
    assert a["artifact_hash"] != b["artifact_hash"]
