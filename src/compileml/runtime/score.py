"""Integer tree-ensemble scoring (ARTIFACT_SPEC.md §4).

The only floating-point operation here is the comparison
``x[feature] <= threshold``, which is exact under IEEE 754. All
accumulation is int64-safe integer addition, so results are
bit-identical on every platform and in every accumulation order.
"""

from __future__ import annotations

from collections.abc import Sequence

from compileml.runtime._intmath import clamp, div_rha

LEAF = -2  # node sentinel: feature[node] == LEAF marks a leaf


def score_micro(model: dict, x: Sequence[float]) -> int:
    """Raw ensemble score in micro units (may fall outside [0, micro_scale])."""
    acc = int(model["base_micro"])
    for tree in model["trees"]:
        feature = tree["feature"]
        threshold = tree["threshold"]
        left = tree["left"]
        right = tree["right"]
        node = 0
        while feature[node] != LEAF:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        acc += tree["value_micro"][node]
    return acc


def latent_from_raw(raw_micro: int, micro_scale: int, scale: int) -> tuple[int, int]:
    """Clamp a raw score and convert to display scale (spec §4).

    Returns (latent_micro, latent_int).
    """
    ratio = micro_scale // scale
    latent_micro = clamp(raw_micro, 0, micro_scale)
    return latent_micro, div_rha(latent_micro, ratio)
