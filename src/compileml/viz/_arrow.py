"""Waterfall arrow-head geometry. Standard library only.

Each waterfall bar is one 7-point polygon — body and head in a single
path, so the head can never detach or mis-scale. The geometry rules here
(dead-zone, adaptive head fraction, direction-aware junction, barb
overhang) are the original waterfall design, ported exactly; the debug
checklist that shipped with the spec lives on as unit tests.

All inputs are **pixel-space** (or a space with a known pixel mapping).
The single most common porting bug is passing the data-space x-range as
``scale_px`` — that collapses every head to ``min_frac``. ``scale_px``
must be the pixel width of the full data range:
``(x_range / view_range) * bar_area_width_px``.
"""

from __future__ import annotations

BAR_H = 18.0  # bar body height, px
HALF_H = BAR_H / 2
OVERHANG = 2.0  # barb overhang beyond the shaft, px — drop it and the
#                 arrow head looks like a flat-ended bar
DEAD_ZONE = 0.3  # px; below this a triangle renders as a smear
TICK_HALF = 3.0  # dead-zone fallback: a 6 px vertical tick


def tri_frac(
    abs_delta_px: float,
    scale_px: float,
    min_frac: float = 0.05,
    max_frac: float = 0.50,
) -> float:
    """Adaptive head fraction: short bars get proportionally chunkier heads.

    Degenerate ``scale_px`` (zero view range) yields ``max_frac`` — visible
    and obviously wrong beats invisible and subtly wrong.
    """
    if scale_px <= 0:
        return max_frac
    raw = max_frac - ((max_frac - min_frac) / scale_px) * abs_delta_px
    return min(max(raw, min_frac), max_frac)


def arrow_points(
    x_start: float,
    x_end: float,
    y_mid: float,
    *,
    half_h: float = HALF_H,
    overhang: float = OVERHANG,
    tri_len: float,
    dead_zone: float = DEAD_ZONE,
) -> list[tuple[float, float]] | None:
    """7-point arrow polygon, direction-aware; ``None`` inside the dead-zone.

    The default dead-zone assumes pixel-space inputs; callers working in a
    different coordinate space (matplotlib data units) must run the pixel
    dead-zone check themselves and pass ``dead_zone=0``.

    Point order is identical for both directions — only the rect/triangle
    junction flips — so the winding stays consistent and the fill never
    self-intersects::

        (xStart, top) (re, top) (re, top-o) (xEnd, yMid) (re, bot+o) (re, bot) (xStart, bot)
    """
    delta = x_end - x_start
    abs_delta = abs(delta)
    if abs_delta < dead_zone:
        return None
    rect_len = abs_delta - tri_len
    junction = x_start + rect_len if delta >= 0 else x_start - rect_len
    top = y_mid - half_h
    bot = y_mid + half_h
    return [
        (x_start, top),
        (junction, top),
        (junction, top - overhang),
        (x_end, y_mid),
        (junction, bot + overhang),
        (junction, bot),
        (x_start, bot),
    ]
