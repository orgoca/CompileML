"""Validation framework and zero-churn recalibration tests."""

import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact, recalibrate_artifact, save_artifact
from compileml.bands import monotone_quantile_bands
from compileml.compile import train_whitebox
from compileml.runtime import decide, verify_artifact
from compileml.validate import validate_artifact

RNG = np.random.default_rng(23)
N, P = 5000, 6
FEATURES = [f"x{i}" for i in range(P)]
REASONS = {
    name: {
        "code": name.upper(),
        "negative": f"{name} raised risk",
        "positive": f"{name} lowered risk",
    }
    for name in FEATURES
}


@pytest.fixture(scope="module")
def fitted():
    X = RNG.standard_normal((N, P))
    teacher = 1.0 / (1.0 + np.exp(-(1.3 * X[:, 0] + 0.9 * X[:, 1] - 0.7 * X[:, 2])))
    y = (RNG.random(N) < teacher).astype(int)
    # The teacher is exactly monotone in x0/x1/x2, so the fixture declares it:
    # every one of the nine checks then runs (check 9 skips when undeclared).
    cst = [1, 1, -1, 0, 0, 0]
    model, _ = train_whitebox(X, teacher, n_estimators=60, random_state=1, monotone_constraints=cst)
    latent = np.clip(model.predict(X), 0, 1)
    spec = monotone_quantile_bands(latent, y, n_bands=8)
    artifact = build_artifact(
        model,
        FEATURES,
        np.median(X, axis=0),
        spec,
        calibration_latent=latent,
        calibration_y=y,
        reasons=REASONS,
        monotone_constraints=cst,
        X_sample=X[:200],
    )
    return X, y, model, latent, artifact


def test_all_checks_pass(fitted):
    X, y, model, latent, artifact = fitted
    report = validate_artifact(
        artifact,
        X_val=X[:1500],
        y_val=y[:1500],
        model=model,
        latent_train=latent,
        require_full_reason_coverage=True,
    )
    assert report["all_pass"], {k: v for k, v in report["checks"].items() if not v["pass"]}
    assert not any(c["skipped"] for c in report["checks"].values())
    assert report["checks"]["8_reason_coverage"]["coverage"] == 1.0
    assert report["checks"]["9_monotone_constraints"]["constrained_features"] == ["x0", "x1", "x2"]


def test_checks_skip_without_inputs(fitted):
    *_, artifact = fitted
    report = validate_artifact(artifact)
    assert report["all_pass"]
    skipped = [k for k, c in report["checks"].items() if c["skipped"]]
    assert "2_reconciliation" in skipped and "6_churn_baseline" in skipped
    assert not report["checks"]["1_integrity"]["skipped"]


def test_integrity_catches_tampering(fitted):
    X, y, model, latent, artifact = fitted
    import copy

    tampered = copy.deepcopy(artifact)
    tampered["bands"]["edges_int"][2] += 1  # hash no longer matches
    report = validate_artifact(tampered)
    assert not report["all_pass"]
    assert not report["checks"]["1_integrity"]["pass"]


def test_monotonicity_fails_on_inverted_outcomes(fitted):
    X, y, model, latent, artifact = fitted
    report = validate_artifact(artifact, X_val=X[:1500], y_val=1 - y[:1500])
    assert not report["checks"]["5_semantic_monotonicity"]["pass"]
    assert not report["all_pass"]


def test_reason_coverage_gate(fitted):
    X, y, model, latent, _ = fitted
    spec = monotone_quantile_bands(latent, y, n_bands=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        partial = build_artifact(
            model,
            FEATURES,
            np.median(X, axis=0),
            spec,
            reasons={"x0": REASONS["x0"]},
        )
    lenient = validate_artifact(partial)
    strict = validate_artifact(partial, require_full_reason_coverage=True)
    assert lenient["checks"]["8_reason_coverage"]["pass"]
    assert not strict["checks"]["8_reason_coverage"]["pass"]
    assert strict["checks"]["8_reason_coverage"]["uncovered_features"] == FEATURES[1:]


def test_validate_from_path(tmp_path, fitted):
    *_, artifact = fitted
    path = tmp_path / "artifact.json"
    save_artifact(artifact, path)
    report = validate_artifact(path)
    assert report["checks"]["1_integrity"]["pass"]


# ------------------------------------------------------------ recalibration
def test_recalibrate_zero_churn(fitted):
    X, y, model, latent, artifact = fitted
    # Fresh outcomes with a higher overall bad rate (portfolio drift).
    rng = np.random.default_rng(99)
    y_new = (rng.random(len(latent)) < np.clip(latent * 1.4, 0, 1)).astype(int)

    new = recalibrate_artifact(artifact, latent, y_new)

    assert verify_artifact(new)
    assert new["artifact_hash"] != artifact["artifact_hash"]
    assert new["metadata"]["recalibration"]["recalibrated_from"] == artifact["artifact_hash"]
    # Model and ladder byte-identical:
    assert new["model"] == artifact["model"]
    assert new["bands"] == artifact["bands"]
    # PD table actually moved:
    assert new["calibration"] != artifact["calibration"]

    # Zero churn: every row keeps its band; PDs shift upward on average.
    old_pd, new_pd = [], []
    for row in X[:400]:
        features = [float(v) for v in row]
        a = decide(artifact, features, explain=False)
        b = decide(new, features, explain=False)
        assert a["band"] == b["band"]
        assert a["latent_int"] == b["latent_int"]
        old_pd.append(a["pd"])
        new_pd.append(b["pd"])
    assert np.mean(new_pd) > np.mean(old_pd)
