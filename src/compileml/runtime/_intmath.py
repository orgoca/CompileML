"""Integer arithmetic primitives (ARTIFACT_SPEC.md §2).

Every rounding decision in a CompileML runtime goes through these two
functions. They use only integer operations, so any conforming target
(COBOL, SQL, Java) can reproduce them bit-for-bit.
"""

from __future__ import annotations


def div_rha(num: int, den: int) -> int:
    """Integer division rounded half away from zero (spec §2.2).

    Computes rha(num / den) for den > 0 using only integer operations.
    """
    if den <= 0:
        raise ValueError("den must be positive")
    if num >= 0:
        return (2 * num + den) // (2 * den)
    return -((-2 * num + den) // (2 * den))


def clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an integer to the inclusive range [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
