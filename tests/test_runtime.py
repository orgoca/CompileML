"""Runtime conformance tests against hand-computed values.

Every expected number in this file was derived by hand from the fixture
in conftest.py — if the runtime disagrees, the runtime is wrong.
"""

import json

import pytest

from compileml.runtime import (
    ArtifactError,
    band_index,
    calibrate_ppm,
    canonical_hash,
    decide,
    load_artifact,
    score_micro,
    verify_artifact,
)
from compileml.runtime._intmath import div_rha

X = [1.0, 5.0, 7.0]  # full=650_000; see hand math below
BASELINE_SCORE = 240_000  # S([0.0, 20.0, 0.0])


# ---------------------------------------------------------------- intmath
@pytest.mark.parametrize(
    ("num", "den", "expected"),
    [
        (1500, 1000, 2),  # 1.5 → away from zero
        (-1500, 1000, -2),
        (1499, 1000, 1),
        (-1499, 1000, -1),
        (2500, 1000, 3),
        (0, 1000, 0),
        (650_000, 1000, 650),
    ],
)
def test_div_rha(num, den, expected):
    assert div_rha(num, den) == expected


# ---------------------------------------------------------------- scoring
def test_score_micro_hand_values(artifact):
    assert score_micro(artifact["model"], X) == 650_000
    assert score_micro(artifact["model"], [0.0, 20.0, 0.0]) == BASELINE_SCORE


def test_score_path_payload(artifact):
    out = decide(artifact, X, explain=False)
    assert out["raw_micro"] == 650_000
    assert out["latent_micro"] == 650_000
    assert out["latent_int"] == 650
    assert out["band"] == "G03"
    assert out["band_idx"] == 2
    assert out["pd_ppm"] == 487_500  # 300_000 + (500_000 * 150_000) / 400_000
    assert out["pd"] == pytest.approx(0.4875)
    assert "reasons_negative" not in out


def test_clamping(artifact):
    # Force a raw score above micro_scale by feeding the all-high path twice over.
    artifact["model"]["base_micro"] = 950_000
    out = decide(artifact, X, explain=False)
    assert out["raw_micro"] == 1_300_000
    assert out["latent_micro"] == 1_000_000
    assert out["latent_int"] == 1000


# ---------------------------------------------------------------- banding
def test_band_boundaries():
    edges = [0, 250, 600, 1000]
    assert band_index(0, edges) == 0
    assert band_index(249, edges) == 0
    assert band_index(250, edges) == 1  # cutoff belongs to the upper band
    assert band_index(599, edges) == 1
    assert band_index(600, edges) == 2
    assert band_index(1000, edges) == 2  # top edge inclusive into last band


# ------------------------------------------------------------- calibration
def test_calibration_bounds_and_interp(artifact):
    cal = artifact["calibration"]
    micro = artifact["model"]["micro_scale"]
    assert calibrate_ppm(50_000, cal, micro) == 20_000  # below first threshold
    assert calibrate_ppm(950_000, cal, micro) == 800_000  # above last
    assert calibrate_ppm(100_000, cal, micro) == 20_000  # exactly at first
    assert calibrate_ppm(650_000, cal, micro) == 487_500  # interior, exact interp
    assert calibrate_ppm(300_000, cal, micro) == 160_000  # midpoint of first segment


def test_calibration_step_mode(artifact):
    cal = dict(artifact["calibration"], mode="step")
    micro = artifact["model"]["micro_scale"]
    assert calibrate_ppm(650_000, cal, micro) == 300_000  # snaps to lower threshold


def test_calibration_absent_is_identity(artifact):
    micro = artifact["model"]["micro_scale"]
    assert calibrate_ppm(650_000, None, micro) == 650_000


# -------------------------------------------------------------- attribution
def test_contributions_hand_values(artifact):
    out = decide(artifact, X, include_contributions=True)
    by_feat = {c["feature"]: c for c in out["contributions"]}
    # Hand-derived: d=[500_000, 60_000, 0]; I01=200_000, I02=0, I12=-50_000
    assert by_feat["f0"]["impact_half_micro"] == 800_000
    assert by_feat["f1"]["impact_half_micro"] == -30_000
    assert by_feat["f2"]["impact_half_micro"] == 50_000
    assert by_feat["f0"]["impact_int"] == 400
    assert by_feat["f1"]["impact_int"] == -15
    assert by_feat["f2"]["impact_int"] == 25


def test_reconciliation_identity_exact(artifact):
    out = decide(artifact, X, include_contributions=True)
    # 2*(full - fbase) == sum(c2) + residual2, exactly, in integers
    lhs = 2 * (out["raw_micro"] - out["baseline_micro"])
    rhs = out["attribution_sum_half_micro"] + out["attribution_residual_half_micro"]
    assert lhs == rhs
    assert out["attribution_residual_half_micro"] == 0  # depth <= 2
    assert out["exact_attribution"] is True


def test_display_impacts_sum_exactly(artifact):
    out = decide(artifact, X, include_contributions=True)
    total = sum(c["impact_int"] for c in out["contributions"])
    ratio = out["micro_scale"] // out["scale"]
    target = div_rha(2 * (out["raw_micro"] - out["baseline_micro"]), 2 * ratio)
    assert total == target == 410


def test_reasons_and_suppression(artifact):
    out = decide(artifact, X)
    # f2 is adverse (c2 = +50_000) but suppressed -> only f0 appears adverse.
    assert [r["feature"] for r in out["reasons_negative"]] == ["f0"]
    assert out["reasons_negative"][0]["code"] == "F0"
    assert out["reasons_negative"][0]["message"] == "f0 hurt you"
    assert out["reasons_negative"][0]["label"] == "Feature Zero"
    assert out["reasons_negative"][0]["direction"] == "risk_increasing"
    # f1 is the only favorable; no dictionary entry -> fallback code/message.
    assert [r["feature"] for r in out["reasons_positive"]] == ["f1"]
    assert out["reasons_positive"][0]["code"] == "POSITIVE_f1"


# ------------------------------------------------------------ missing values
def test_missing_policy_baseline(artifact):
    out = decide(artifact, [None, 5.0, 7.0], explain=False)
    # f0 -> baseline 0.0: TreeA left (-100k), TreeB left-left (-50k): 150_000
    assert out["raw_micro"] == 150_000
    assert out["band"] == "G01"


def test_missing_policy_nan_treated_as_missing(artifact):
    out = decide(artifact, [float("nan"), 5.0, 7.0], explain=False)
    assert out["raw_micro"] == 150_000


def test_missing_policy_reject(artifact):
    artifact["features"]["missing_policy"] = "reject"
    with pytest.raises(ArtifactError, match="missing value"):
        decide(artifact, [None, 5.0, 7.0], explain=False)


def test_wrong_feature_count(artifact):
    with pytest.raises(ArtifactError, match="expected 3 features"):
        decide(artifact, [1.0, 2.0], explain=False)


# ------------------------------------------------------------- hash & load
def test_hash_roundtrip(tmp_path, artifact):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    loaded = load_artifact(path)
    assert loaded["artifact_hash"] == artifact["artifact_hash"]
    # payloads include elapsed_ms; compare the decision fields instead:
    a = decide(loaded, X, explain=False)
    b = decide(artifact, X, explain=False)
    for key in ("band", "latent_int", "pd_ppm", "raw_micro"):
        assert a[key] == b[key]


def test_tampered_artifact_refused(tmp_path, artifact):
    artifact["bands"]["edges_int"] = [0, 260, 600, 1000]  # tamper after hashing
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        load_artifact(path)
    assert not verify_artifact(artifact)


def test_verify_can_be_disabled(tmp_path, artifact):
    artifact["metadata"]["note"] = "modified"
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    loaded = load_artifact(path, verify=False)
    assert loaded["metadata"]["note"] == "modified"


def test_canonical_hash_ignores_key_order(artifact):
    reordered = dict(reversed(list(artifact.items())))
    assert canonical_hash(reordered) == canonical_hash(artifact)


# ------------------------------------------------------ runtime self-check
def test_decide_refuses_nonreconciling_explanation(artifact, monkeypatch):
    """A corrupted attribution on an exact-attribution artifact must be
    refused, not emitted. Forced here by stubbing the contribution kernel."""
    import sys

    # The package attribute `decide` is the function (re-exported by
    # __init__), which shadows the submodule in `import … as` — go through
    # sys.modules for the module object itself.
    decide_module = sys.modules["compileml.runtime.decide"]
    real = decide_module.contributions_half_micro

    def corrupted(model, x, baseline):
        c2, full, fbase, _ = real(model, x, baseline)
        return [v + 2 for v in c2], full, fbase, 6  # nonzero residual, claims intact

    monkeypatch.setattr(decide_module, "contributions_half_micro", corrupted)
    with pytest.raises(ArtifactError, match="refusing to emit"):
        decide(artifact, X)

    # Depth>2 artifacts legitimately carry residuals: no refusal there.
    artifact["runtime"]["exact_attribution"] = False
    out = decide(artifact, X)
    assert out["attribution_residual_half_micro"] == 6


# ---------------------------------------------------------- input precision
def test_input_precision_float32_changes_routing(artifact):
    """A float64 input just above a float32 threshold routes differently
    once the artifact demands binary32 input quantization (spec §4)."""
    import struct

    thr_f32 = struct.unpack("<f", struct.pack("<f", 0.1))[0]  # 0.10000000149...
    x0 = 0.100000002  # > thr in float64, rounds down to thr in float32

    artifact["model"]["trees"] = [artifact["model"]["trees"][0]]
    artifact["model"]["trees"][0]["threshold"][0] = thr_f32

    artifact["model"]["input_precision"] = "float64"
    right = decide(artifact, [x0, 20.0, 0.0], explain=False)
    artifact["model"]["input_precision"] = "float32"
    left = decide(artifact, [x0, 20.0, 0.0], explain=False)

    assert right["raw_micro"] == 300_000 + 200_000  # took the right branch
    assert left["raw_micro"] == 300_000 - 100_000  # f32 quantization: left branch
