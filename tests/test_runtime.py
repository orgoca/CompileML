"""Runtime conformance tests against hand-computed values.

Every expected number in this file was derived by hand from the fixture
in conftest.py — if the runtime disagrees, the runtime is wrong.
"""

import json
import warnings

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


# ------------------------------------------------- two derivations, one answer
#
# Attribution is now aggregated per tree rather than by perturbing one feature
# at a time. Integer addition is associative, so the regrouping must be
# bit-identical rather than merely close. The perturbation derivation is kept
# precisely so that claim can be checked rather than asserted (#15).


def _random_model(rng, n_trees, n_features, max_depth):
    """A small quantized ensemble with the shape the runtime expects."""
    import numpy as np

    from compileml.compile import extract_trees, quantize_model, train_whitebox

    X = rng.standard_normal((600, n_features))
    latent = 1 / (1 + np.exp(-(1.1 * X[:, 0] + 0.7 * X[:, 1 % n_features])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = train_whitebox(
            X,
            latent,
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=int(rng.integers(1e6)),
        )
    return quantize_model(extract_trees(model)), X


@pytest.mark.parametrize(
    "n_trees,n_features,max_depth",
    [(5, 3, 1), (12, 8, 2), (30, 20, 2), (8, 12, 3), (3, 40, 2)],
)
def test_per_tree_attribution_matches_the_perturbation_derivation(n_trees, n_features, max_depth):
    """Same integers by a different route, across shapes and depths."""
    import numpy as np

    from compileml.runtime.explain import (
        contributions_half_micro,
        contributions_half_micro_reference,
    )

    rng = np.random.default_rng(n_trees * 1000 + n_features * 10 + max_depth)
    model, X = _random_model(rng, n_trees, n_features, max_depth)
    baseline = [float(v) for v in np.median(X, axis=0)]

    for i in range(20):
        row = [float(v) for v in X[i]]
        fast = contributions_half_micro(model, row, baseline)
        ref = contributions_half_micro_reference(model, row, baseline)
        assert fast == ref, f"row {i} disagrees at depth {max_depth}"


def test_attribution_cost_does_not_grow_with_feature_count():
    """The point of #15: work is set by trees, not by how wide the model is.

    Counts leaf traversals rather than timing, so it cannot flake on a busy
    machine.
    """
    import numpy as np

    from compileml.runtime import explain as explain_mod

    calls = {"n": 0}
    original = explain_mod._tree_leaf

    def counting(tree, x):
        calls["n"] += 1
        return original(tree, x)

    counts = {}
    for n_features in (5, 40):
        rng = np.random.default_rng(7)
        model, X = _random_model(rng, 10, n_features, 2)
        baseline = [float(v) for v in np.median(X, axis=0)]
        calls["n"] = 0
        explain_mod._tree_leaf = counting
        try:
            explain_mod.contributions_half_micro(model, [float(v) for v in X[0]], baseline)
        finally:
            explain_mod._tree_leaf = original
        counts[n_features] = calls["n"]

    # Eight times the feature count changes the work by a few percent. It is
    # not identical because tree *structure* varies — a tree splitting on two
    # features costs four lookups, one splitting on three costs eight — but it
    # does not scale with p. The perturbation derivation would need
    # 1 + p + p(p-1)/2 ensemble traversals: 16 at five features, 821 at forty.
    assert counts[40] < counts[5] * 1.25, counts
    reference_traversals = {p: 1 + p + p * (p - 1) // 2 for p in (5, 40)}
    assert reference_traversals[40] > reference_traversals[5] * 25
