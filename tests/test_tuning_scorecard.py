"""Tests for the tuning sweeps, band efficiency, and the exact scorecard."""

import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact
from compileml.bands import band_efficiency, monotone_quantile_bands
from compileml.compile import train_whitebox
from compileml.runtime import decide, score_micro
from compileml.scorecard import (
    build_scorecard,
    score_from_scorecard,
    scorecard_to_csv,
    scorecard_to_markdown,
)
from compileml.tune import sweep_bands, sweep_whitebox

RNG = np.random.default_rng(19)
N, P = 4000, 6
FEATURES = [f"f{i}" for i in range(P)]


@pytest.fixture(scope="module")
def data():
    X = RNG.standard_normal((N, P))
    teacher = 1.0 / (1.0 + np.exp(-(1.4 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3])))
    y = (RNG.random(N) < teacher).astype(int)
    return X, teacher, y


def _artifact_for(model, X, y):
    latent = np.clip(model.predict(X), 0, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_artifact(
            model,
            FEATURES,
            np.median(X, axis=0),
            monotone_quantile_bands(latent, y, n_bands=6),
            calibration_latent=latent,
            calibration_y=y,
        )


# ------------------------------------------------------------------ sweeps
def test_sweep_whitebox_shape_and_semantics(data):
    X, teacher, y = data
    rows = sweep_whitebox(X, teacher, y, trees_grid=(10, 30), depth_grid=(1, 2, 3), random_state=0)
    assert len(rows) == 6
    by_key = {(r["n_estimators"], r["max_depth"]): r for r in rows}
    assert by_key[(10, 1)]["exact_attribution"] is True
    assert by_key[(30, 2)]["exact_attribution"] is True
    assert by_key[(30, 3)]["exact_attribution"] is False
    # More trees at fixed depth: fidelity should not get worse in-sample.
    assert by_key[(30, 2)]["spearman_vs_teacher"] >= by_key[(10, 2)]["spearman_vs_teacher"]
    # Size grows with trees; explain cost was actually measured.
    assert by_key[(30, 2)]["model_kb"] > by_key[(10, 2)]["model_kb"]
    assert all(r["explain_ms_per_row"] >= 0 for r in rows)
    assert all(r["in_sample"] for r in rows)


def test_sweep_whitebox_holdout_flag(data):
    X, teacher, y = data
    rows = sweep_whitebox(
        X[:3000],
        teacher[:3000],
        y[:3000],
        trees_grid=(10,),
        depth_grid=(2,),
        X_val=X[3000:],
        y_val=y[3000:],
        teacher_latent_val=teacher[3000:],
    )
    assert rows[0]["in_sample"] is False


def test_sweep_bands_monotone_gain(data):
    X, teacher, y = data
    model, _ = train_whitebox(X, teacher, n_estimators=60, random_state=0)
    latent = np.clip(model.predict(X), 0, 1)
    rows = sweep_bands(latent, y, k_grid=(4, 8, 12), n_boot=40)
    assert [r["n_bands"] for r in rows] == [4, 8, 12]
    # More bands => the ladder can only keep more of the continuous Gini.
    ginis = [r["band_ordinal_gini"] for r in rows]
    assert ginis[0] <= ginis[-1] + 1e-9
    assert all(r["gini_gap"] >= -1e-9 for r in rows)
    assert all(r["int_edge_collisions"] == 0 for r in rows)


# --------------------------------------------------------- band efficiency
def test_band_efficiency_coarse_bands_leave_money(data):
    X, teacher, y = data
    model, _ = train_whitebox(X, teacher, n_estimators=60, random_state=0)
    latent = np.clip(model.predict(X), 0, 1)

    coarse = band_efficiency(latent, y, monotone_quantile_bands(latent, y, n_bands=2), n_boot=60)
    fine = band_efficiency(latent, y, monotone_quantile_bands(latent, y, n_bands=12), n_boot=60)
    assert coarse["gini_gap"] > fine["gini_gap"]  # two bands waste more
    # With 2 bands over a strong latent, at least one band is refinable.
    verdicts = {p["verdict"] for p in coarse["per_band"]}
    assert "refinable" in verdicts
    assert coarse["worst_band"] is not None


def test_band_efficiency_accepts_artifact(data):
    X, teacher, y = data
    model, _ = train_whitebox(X, teacher, n_estimators=40, random_state=0)
    artifact = _artifact_for(model, X, y)
    latent = np.clip(model.predict(X), 0, 1)
    eff = band_efficiency(latent, y, artifact, n_boot=40)
    assert len(eff["per_band"]) == len(artifact["bands"]["labels"])
    assert 0 <= eff["band_ordinal_gini"] <= eff["continuous_gini"] + 1e-9


def test_validate_check4_gains_efficiency_fields(data):
    from compileml.validate import validate_artifact

    X, teacher, y = data
    model, _ = train_whitebox(X, teacher, n_estimators=40, random_state=0)
    artifact = _artifact_for(model, X, y)

    report = validate_artifact(artifact, X_val=X[:1500], y_val=y[:1500])
    check4 = report["checks"]["4_band_properties"]
    assert "banding_gini_gap" in check4 and "worst_within_band_auc" in check4
    assert check4["pass"]  # advisory by default

    # Gate it hard: an impossible threshold must fail the check.
    strict = validate_artifact(artifact, X_val=X[:1500], y_val=y[:1500], max_within_band_auc=0.50)
    assert not strict["checks"]["4_band_properties"]["pass"]


# --------------------------------------------------------------- scorecard
def test_scorecard_depth1_resums_exactly(data):
    X, teacher, y = data
    model, _ = train_whitebox(X, teacher, n_estimators=40, max_depth=1, random_state=0)
    artifact = _artifact_for(model, X, y)
    scorecard = build_scorecard(artifact)

    assert scorecard["interactions"] == {}  # depth 1: pure classic scorecard
    assert scorecard["main_effects"]  # at least one feature was used
    for row in X[:300]:
        row_f = [float(v) for v in row]
        assert score_from_scorecard(scorecard, row_f) == score_micro(artifact["model"], row_f)


def test_scorecard_depth2_resums_exactly_with_interactions(data):
    X, teacher, y = data
    model, _ = train_whitebox(X, teacher, n_estimators=60, max_depth=2, random_state=0)
    artifact = _artifact_for(model, X, y)
    scorecard = build_scorecard(artifact)

    assert scorecard["interactions"]  # the teacher has an interaction term
    for row in X[:300]:
        row_f = [float(v) for v in row]
        assert score_from_scorecard(scorecard, row_f) == score_micro(artifact["model"], row_f)
    # And therefore agrees with the full runtime's raw score.
    out = decide(artifact, [float(v) for v in X[0]], explain=False)
    assert score_from_scorecard(scorecard, [float(v) for v in X[0]]) == out["raw_micro"]


def test_scorecard_refuses_depth3(data):
    X, teacher, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deep, _ = train_whitebox(X, teacher, n_estimators=20, max_depth=3, random_state=0)
    artifact = _artifact_for(deep, X, y)
    with pytest.raises(ValueError, match="no exact scorecard exists"):
        build_scorecard(artifact)


def test_scorecard_renderers(data):
    X, teacher, y = data
    deep2, _ = train_whitebox(X, teacher, n_estimators=40, max_depth=2, random_state=0)
    card2 = build_scorecard(_artifact_for(deep2, X, y))

    md = scorecard_to_markdown(card2, labels={"f0": "Utilization"})
    assert "# Scorecard" in md and "Utilization" in md
    assert "re-sum" in md
    if card2["interactions"]:
        assert "Interaction:" in md

    # Depth-1 guarantees main-effect rows (a stump cannot be an interaction).
    deep1, _ = train_whitebox(X, teacher, n_estimators=30, max_depth=1, random_state=0)
    card1 = build_scorecard(_artifact_for(deep1, X, y))
    csv_text = scorecard_to_csv(card1)
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("kind,feature,interval")
    assert any(line.startswith("base,") for line in lines)
    assert any(line.startswith("main,") for line in lines)
    # Row count is exactly base + every bin + every grid cell.
    expected = 1 + sum(len(e["bins"]) for e in card1["main_effects"].values())
    assert len(lines) == 1 + expected  # header + rows


def test_scorecard_cli(tmp_path, data):
    from compileml.artifact import save_artifact
    from compileml.cli import main

    X, teacher, y = data
    model, _ = train_whitebox(X, teacher, n_estimators=30, max_depth=2, random_state=0)
    artifact = _artifact_for(model, X, y)
    path = tmp_path / "artifact.json"
    save_artifact(artifact, path)

    out = tmp_path / "scorecard.md"
    assert main(["scorecard", str(path), "--out", str(out)]) == 0
    assert "# Scorecard" in out.read_text(encoding="utf-8")
