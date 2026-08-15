"""Cross-platform determinism oracle.

The reference artifact and its expected outputs were generated once and are
committed. Every CI matrix cell — every OS, every Python version — must
reproduce these integers exactly. If this test fails on some platform, the
determinism claim is broken and the failure is a release blocker, not a flake.
"""

import json
from pathlib import Path

import pytest

from compileml.runtime import decide, load_artifact

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def reference():
    artifact = load_artifact(DATA / "reference_artifact.json")  # hash-verified
    with open(DATA / "reference_expected.json", encoding="utf-8") as f:
        expected = json.load(f)
    return artifact, expected


def test_reference_hash():
    artifact = load_artifact(DATA / "reference_artifact.json")
    assert artifact["artifact_hash"] == (
        "86b5fd378ca6a86d2d596d0cebdc1d6171ab292718328ca088fc4423941ba7ed"
    )


def test_reference_scores_exact(reference):
    artifact, expected = reference
    for row, want in zip(expected["rows"], expected["scores"]):
        out = decide(artifact, row, explain=False)
        assert out["raw_micro"] == want["raw_micro"]
        assert out["latent_int"] == want["latent_int"]
        assert out["band"] == want["band"]
        assert out["pd_ppm"] == want["pd_ppm"]


def test_reference_explanations_exact(reference):
    artifact, expected = reference
    for row, want in zip(expected["rows"], expected["explains"]):
        out = decide(artifact, row, include_contributions=True)
        assert [c["impact_int"] for c in out["contributions"]] == want["impact_int"]
        assert out["attribution_residual_half_micro"] == want["residual2"]
        assert [r["code"] for r in out["reasons_negative"]] == want["reasons_negative"]
        assert [r["code"] for r in out["reasons_positive"]] == want["reasons_positive"]
