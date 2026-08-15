"""Dependency-free SVG waterfall — standard library only.

The decision payload is plain integers, so rendering it needs no plotting
stack at all. Useful for docs, emails, and audit records; also mildly
on-brand: even the chart has no runtime dependencies. Output is
deterministic — same payload, same bytes.
"""

from __future__ import annotations

from html import escape

from compileml.viz._data import waterfall_segments

_FILL = {
    "up": "#e67e22",
    "down": "#2ecc71",
    "remainder": "#95a5a6",
    "residual": "#8e44ad",
    "base": "#111111",
}
ROW_H = 30
LABEL_W = 210
PAD = 14


def _fill(seg: dict) -> str:
    if seg["kind"] == "remainder":
        return _FILL["remainder"]
    if seg["kind"] == "residual":
        return _FILL["residual"]
    return _FILL["up"] if seg["half_micro"] > 0 else _FILL["down"]


def waterfall_svg(
    decision: dict, *, max_features: int = 10, labels: dict | None = None, width: int = 760
) -> str:
    """Render one decision's waterfall as a self-contained SVG string."""
    data = waterfall_segments(decision, max_features=max_features, labels=labels)
    micro = data["micro_scale"]
    base = data["baseline_micro"] / micro
    final = data["raw_micro"] / micro
    segments = data["segments"]

    # Latent-space extent of the walk.
    cursor, points = base, [base, final]
    for seg in segments:
        cursor += seg["latent_delta"]
        points.append(cursor)
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    lo -= 0.06 * span
    hi += 0.06 * span

    plot_left, plot_right = LABEL_W, width - PAD

    def x(value: float) -> float:
        return plot_left + (value - lo) / (hi - lo) * (plot_right - plot_left)

    n_rows = len(segments) + 2
    height = PAD + 26 + n_rows * ROW_H + PAD
    parts: list[str] = []
    put = parts.append

    put(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    )
    put(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    title = f"Band {escape(str(data['band']))} · latent_int {data['latent_int']}"
    if data.get("pd") is not None:
        title += f" · PD {data['pd']:.4f}"
    put(
        f'<text x="{PAD}" y="{PAD + 12}" font-size="13" font-weight="bold" '
        f'fill="#111">{title}</text>'
    )

    top = PAD + 26
    baseline_x = x(base)
    put(
        f'<line x1="{baseline_x:.1f}" y1="{top}" x2="{baseline_x:.1f}" '
        f'y2="{top + n_rows * ROW_H - 8}" stroke="#999" stroke-dasharray="3,3" stroke-width="1"/>'
    )

    def row_text(row: int, label: str, bold: bool = False) -> None:
        weight = ' font-weight="bold"' if bold else ""
        put(
            f'<text x="{LABEL_W - 8}" y="{top + row * ROW_H + 19}" font-size="11" '
            f'text-anchor="end" fill="#333"{weight}>{escape(label)}</text>'
        )

    # Baseline row.
    row_text(0, "Baseline")
    put(f'<circle cx="{baseline_x:.1f}" cy="{top + 15}" r="4" fill="{_FILL["base"]}"/>')
    put(
        f'<text x="{baseline_x + 7:.1f}" y="{top + 19}" font-size="10" fill="#111">'
        f"{base:.4f}</text>"
    )

    cursor = base
    for i, seg in enumerate(segments, start=1):
        row_text(i, seg["label"])
        delta = seg["latent_delta"]
        x0, x1 = x(cursor), x(cursor + delta)
        left, bar_width = (x0, x1 - x0) if x1 >= x0 else (x1, x0 - x1)
        put(
            f'<rect x="{left:.1f}" y="{top + i * ROW_H + 4}" width="{max(bar_width, 1.0):.1f}" '
            f'height="20" fill="{_fill(seg)}" rx="2"/>'
        )
        note = f"{delta:+.4f}"
        if seg["impact_int"] is not None:
            note += f" ({seg['impact_int']:+d})"
        anchor, side = (x1 + 6, "start") if delta >= 0 else (x1 - 6, "end")
        put(
            f'<text x="{anchor:.1f}" y="{top + i * ROW_H + 18}" font-size="10" '
            f'text-anchor="{side}" fill="#333">{note}</text>'
        )
        cursor += delta

    # Score row.
    row = len(segments) + 1
    row_text(row, "Score", bold=True)
    x_final = x(final)
    put(
        f'<rect x="{min(x(lo + 0.0), x_final):.1f}" y="{top + row * ROW_H + 4}" '
        f'width="{abs(x_final - plot_left):.1f}" height="20" fill="{_FILL["base"]}" rx="2"/>'
    )
    put(
        f'<text x="{x_final + 7:.1f}" y="{top + row * ROW_H + 18}" font-size="10" '
        f'font-weight="bold" fill="#111">{final:.4f}</text>'
    )
    put("</svg>")
    return "".join(parts)
