"""Band assignment on the fixed-point ladder (ARTIFACT_SPEC.md §5).

This is the single band-assignment algorithm. There is deliberately no
float-space variant: bands are defined on integer edges, and every
consumer — Python, COBOL, SQL — walks the same integers.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence


def band_index(latent_int: int, edges_int: Sequence[int]) -> int:
    """Index of the band containing ``latent_int``.

    Bands are left-closed, right-open on the interior cutoffs; values below
    the first edge belong to band 0 and values at or above the last edge
    belong to the final band.
    """
    return bisect_right(edges_int[1:-1], latent_int)


def band_label(latent_int: int, bands: dict) -> tuple[int, str]:
    """(index, label) for the band containing ``latent_int``."""
    idx = band_index(latent_int, bands["edges_int"])
    return idx, str(bands["labels"][idx])
