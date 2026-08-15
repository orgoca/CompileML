"""Shared fixtures: a small hand-built artifact whose numbers are verifiable by hand."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compileml.runtime.io import canonical_hash  # noqa: E402

LEAF = -2


def _stump(feature: int, threshold: float, left_micro: int, right_micro: int) -> dict:
    """A depth-1 tree: one split, two leaves."""
    return {
        "feature": [feature, LEAF, LEAF],
        "threshold": [threshold, 0.0, 0.0],
        "left": [1, -1, -1],
        "right": [2, -1, -1],
        "value_micro": [0, left_micro, right_micro],
    }


@pytest.fixture
def artifact() -> dict:
    """Three features, one stump + one depth-2 tree, 3 bands, linear calibration.

    micro_scale=1_000_000, scale=1000, ratio=1000, base_micro=300_000.
    """
    doc = {
        "artifact_type": "compileml.decision_artifact",
        "schema_version": 2,
        "scale": 1000,
        "model": {
            "kind": "tree_ensemble_int",
            "micro_scale": 1_000_000,
            "base_micro": 300_000,
            "n_features": 3,
            "trees": [
                # Tree A: split on f0 at 0.5 → -100_000 / +200_000
                _stump(0, 0.5, -100_000, 200_000),
                # Tree B: root on f1 at 10.0; left child splits on f0 at 0.5
                # (→ interaction f0×f1), right child splits on f2 at 5.0.
                {
                    "feature": [1, 0, LEAF, LEAF, 2, LEAF, LEAF],
                    "threshold": [10.0, 0.5, 0.0, 0.0, 5.0, 0.0, 0.0],
                    "left": [1, 2, -1, -1, 5, -1, -1],
                    "right": [4, 3, -1, -1, 6, -1, -1],
                    "value_micro": [0, 0, -50_000, 150_000, 0, 40_000, 90_000],
                },
            ],
        },
        "calibration": {
            "mode": "linear_int",
            "f_micro": [100_000, 500_000, 900_000],
            "pd_ppm": [20_000, 300_000, 800_000],
        },
        "bands": {
            "edges_int": [0, 250, 600, 1000],
            "labels": ["G01", "G02", "G03"],
            "boundary": "left_closed_right_open",
        },
        "features": {
            "names": ["f0", "f1", "f2"],
            "baseline": [0.0, 20.0, 0.0],
            "missing_policy": "baseline",
            "display_names": {"f0": "Feature Zero"},
            "meta": [],
        },
        "reasons": {
            "f0": {"code": "F0", "negative": "f0 hurt you", "positive": "f0 helped you"},
            "f2": {
                "code": "F2",
                "negative": "f2 hurt you",
                "positive": "f2 helped you",
                "suppress": True,
            },
        },
        "runtime": {
            "attribution": "pairwise_interaction_int",
            "top_k": 5,
            "whitebox_max_depth": 2,
            "exact_attribution": True,
        },
        "metadata": {"origin": "hand-built test fixture"},
    }
    doc["artifact_hash"] = canonical_hash(doc)
    return doc
