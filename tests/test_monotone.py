"""Monotone constraints: HGB backend, trainer-independent verification, check 9.

The teacher used here has a deliberate *local inversion* on f0
(``x0 - 1.2·sin(2.5·x0)`` dips while trending up), so an unconstrained
whitebox provably learns a non-monotone shape — the verifier must catch
it — while the constrained backend must not.
"""

import json
import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact
from compileml.compile import (
    extract_trees,
    normalize_constraints,
    quantize_model,
    score_float,
    scorecard_monotone_report,
    train_whitebox,
    verify_monotone_constraints,
)
from compileml.runtime import decide
from compileml.runtime.io import canonical_hash
from compileml.scorecard import build_scorecard
from compileml.validate import validate_artifact

RNG = np.random.default_rng(7)
N, P = 3000, 4
FEATURES = ["util", "dpd", "tenure", "inq"]
CONSTRAINTS = {"util": 1}  # by name, resolved at build time
EDGES = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.fixture(scope="module")
def data():
    X = RNG.standard_normal((N, P))
    # Locally inverted in x0: monotone overall trend, non-monotone shape.
    latent = 1.0 / (1.0 + np.exp(-(X[:, 0] - 1.2 * np.sin(2.5 * X[:, 0]) + 0.5 * X[:, 1])))
    y = (RNG.random(N) < latent).astype(int)
    return X, latent, y


@pytest.fixture(scope="module")
def constrained_model(data):
    X, latent, _ = data
    model, metrics = train_whitebox(
        X, latent, n_estimators=25, max_depth=2, monotone_constraints={0: 1}
    )
    assert type(model).__name__ == "HistGradientBoostingRegressor"
    # The constraint forbids tracking the teacher's deliberate dips, so
    # fidelity sits below the unconstrained fit — a floor, not a claim.
    assert metrics["spearman"] > 0.85
    return model


@pytest.fixture(scope="module")
def unconstrained_model(data):
    X, latent, _ = data
    model, _ = train_whitebox(X, latent, n_estimators=25, max_depth=2)
    assert type(model).__name__ == "GradientBoostingRegressor"
    return model


def _build(model, data, **kwargs):
    X, latent, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # reason coverage warned, tested elsewhere
        return build_artifact(
            model,
            FEATURES,
            np.median(X, axis=0),
            EDGES,
            calibration_latent=latent,
            calibration_y=y,
            X_sample=X[:200],
            **kwargs,
        )


@pytest.fixture(scope="module")
def constrained_artifact(constrained_model, data):
    return _build(constrained_model, data, monotone_constraints=CONSTRAINTS)


# ---------------------------------------------------------- normalization
def test_normalize_constraint_forms():
    assert normalize_constraints([0, 1, 0, -1], 4) == [0, 1, 0, -1]
    assert normalize_constraints({1: 1, 3: -1}, 4) == [0, 1, 0, -1]
    assert normalize_constraints({"b": 1}, 3, feature_names=["a", "b", "c"]) == [0, 1, 0]
    assert normalize_constraints(None, 4) is None
    assert normalize_constraints([0, 0, 0, 0], 4) is None  # all-zeros = unconstrained


def test_normalize_constraint_rejects():
    with pytest.raises(ValueError, match="length"):
        normalize_constraints([1, 0], 4)
    with pytest.raises(ValueError, match="signs"):
        normalize_constraints([1, 2, 0, 0], 4)
    with pytest.raises(ValueError, match="out of range"):
        normalize_constraints({7: 1}, 4)
    with pytest.raises(ValueError, match="unknown feature name"):
        normalize_constraints({"nope": 1}, 3, feature_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="no feature names"):
        normalize_constraints({"util": 1}, 4)


# ------------------------------------------------------------- extraction
def test_hist_extraction_parity(constrained_model, data):
    X, _, _ = data
    extracted = extract_trees(constrained_model)
    assert extracted.family == "sklearn_hist"
    assert extracted.learning_rate == 1.0  # HGB leaves arrive pre-shrunk
    for row in X[:50]:
        assert score_float(extracted, [float(v) for v in row]) == pytest.approx(
            float(constrained_model.predict(row.reshape(1, -1))[0]), abs=1e-9
        )


def test_hist_categorical_splits_rejected(data):
    from sklearn.ensemble import HistGradientBoostingRegressor

    X, latent, _ = data
    Xc = X.copy()
    Xc[:, 3] = RNG.integers(0, 4, N)
    model = HistGradientBoostingRegressor(max_iter=3, categorical_features=[3])
    model.fit(Xc, latent)
    with pytest.raises(ValueError, match="categorical"):
        extract_trees(model)


# ---------------------------------------------- verification has teeth
def test_verifier_catches_unconstrained_inversion(unconstrained_model):
    """The inversion teacher makes an unconstrained whitebox non-monotone."""
    model_int = quantize_model(extract_trees(unconstrained_model))
    report = verify_monotone_constraints(model_int, [1, 0, 0, 0])
    assert not report["ok"]
    assert report["n_violations"] > 0
    v = report["violations"][0]
    assert v["feature"] == 0 and v["sign"] == 1
    # The reported leaf sequence really does decrease somewhere.
    seq = v["leaf_values_micro"]
    assert any(b < a for a, b in zip(seq, seq[1:]))


def test_verifier_passes_constrained_model(constrained_model):
    model_int = quantize_model(extract_trees(constrained_model))
    report = verify_monotone_constraints(model_int, [1, 0, 0, 0])
    assert report["ok"] and report["n_violations"] == 0


def test_build_refuses_violating_declaration(unconstrained_model, data):
    with pytest.raises(ValueError, match="monotone constraint violated"):
        _build(unconstrained_model, data, monotone_constraints=CONSTRAINTS)


# ------------------------------------------------------------ the artifact
def test_constraints_recorded_and_hashed(constrained_artifact):
    assert constrained_artifact["model"]["monotone_constraints"] == [1, 0, 0, 0]
    # Hash-covered: flipping the recorded sign breaks verification.
    tampered = json.loads(json.dumps(constrained_artifact))
    tampered["model"]["monotone_constraints"] = [-1, 0, 0, 0]
    assert canonical_hash(tampered) != constrained_artifact["artifact_hash"]


def test_constrained_rebuild_hash_identical(constrained_model, data):
    """Same data, fresh fit + build → byte-identical artifact (determinism)."""
    X, latent, _ = data
    model2, _ = train_whitebox(X, latent, n_estimators=25, max_depth=2, monotone_constraints={0: 1})
    art2 = _build(model2, data, monotone_constraints=CONSTRAINTS)
    art1 = _build(constrained_model, data, monotone_constraints=CONSTRAINTS)
    assert art1["artifact_hash"] == art2["artifact_hash"]


def test_decide_runs_on_constrained_artifact(constrained_artifact, data):
    X, _, _ = data
    out = decide(constrained_artifact, [float(v) for v in X[0]])
    assert out["exact_attribution"] is True
    assert out["attribution_residual_half_micro"] == 0


# -------------------------------------------------------------- check 9
def test_check9_passes_and_names_features(constrained_artifact):
    result = validate_artifact(constrained_artifact)
    c9 = result["checks"]["9_monotone_constraints"]
    assert c9["pass"] and not c9["skipped"]
    assert c9["constrained_features"] == ["util"]


def test_check9_skips_without_declaration(unconstrained_model, data):
    art = _build(unconstrained_model, data)
    c9 = validate_artifact(art)["checks"]["9_monotone_constraints"]
    assert c9["pass"] and c9["skipped"]


def test_check9_fails_on_tampered_declaration(artifact):
    """Hand-built fixture (conftest) is increasing in f0; declare the opposite."""
    doc = json.loads(json.dumps(artifact))
    del doc["artifact_hash"]
    doc["model"]["monotone_constraints"] = [-1, 0, 0]
    doc["artifact_hash"] = canonical_hash(doc)
    result = validate_artifact(doc)
    c9 = result["checks"]["9_monotone_constraints"]
    assert not c9["pass"] and not result["all_pass"]
    assert c9["n_violations"] > 0

    doc["model"]["monotone_constraints"] = [1, 0, 0]
    del doc["artifact_hash"]
    doc["artifact_hash"] = canonical_hash(doc)
    assert validate_artifact(doc)["checks"]["9_monotone_constraints"]["pass"]


# ------------------------------------------------- scorecard aggregate
def test_scorecard_aggregate_report(constrained_artifact):
    scorecard = build_scorecard(constrained_artifact)
    report = scorecard_monotone_report(scorecard, CONSTRAINTS)
    assert report["ok"]
    assert report["per_feature"]["util"]["ok"]
    assert report["per_feature"]["util"]["worst_increment_micro"] >= 0


def test_scorecard_aggregate_catches_inversion(unconstrained_model, data):
    art = _build(unconstrained_model, data)
    scorecard = build_scorecard(art)
    report = scorecard_monotone_report(scorecard, CONSTRAINTS)
    assert not report["ok"]
    assert report["per_feature"]["util"]["worst_increment_micro"] < 0


# --------------------------------------------------------- name plumbing
def test_train_whitebox_name_dict_needs_names(data):
    X, latent, _ = data
    with pytest.raises(ValueError, match="no feature names"):
        train_whitebox(X, latent, n_estimators=2, monotone_constraints={"util": 1})


def test_train_whitebox_accepts_dataframe_names(data):
    pd = pytest.importorskip("pandas")
    X, latent, _ = data
    model, _ = train_whitebox(
        pd.DataFrame(X, columns=FEATURES),
        latent,
        n_estimators=2,
        monotone_constraints={"util": 1},
    )
    assert type(model).__name__ == "HistGradientBoostingRegressor"
