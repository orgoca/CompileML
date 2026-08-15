"""Matplotlib renderings of decision payloads (``compileml[viz]``).

Every function takes ``decide()`` output — the plots draw the deployed
integers, they never recompute them. See ``_data.py`` for the exact
segment/table construction these renderers share with the SVG exporter.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from compileml.viz._data import band_table, driver_table, waterfall_segments

COLORS = {
    "up": "#e67e22",  # risk-increasing
    "down": "#2ecc71",  # risk-decreasing
    "base": "#111111",
    "remainder": "#95a5a6",
    "residual": "#8e44ad",
    "good": "#2ecc71",
    "bad": "#e67e22",
    "neutral": "#7f8c8d",
}


def _segment_color(seg: dict, colors: dict) -> str:
    if seg["kind"] == "remainder":
        return colors["remainder"]
    if seg["kind"] == "residual":
        return colors["residual"]
    return colors["up"] if seg["half_micro"] > 0 else colors["down"]


def waterfall(
    decision: dict,
    *,
    max_features: int = 10,
    labels: dict | None = None,
    colors: dict | None = None,
    ax=None,
    title: str | None = None,
):
    """Single-decision waterfall: baseline → per-feature impacts → score.

    The bars sum exactly to the decision (spec §7.4); what you see is the
    audit, drawn. Requires a payload from
    ``decide(..., include_contributions=True)``.

    Returns:
        (figure, axes)
    """
    colors = {**COLORS, **(colors or {})}
    data = waterfall_segments(decision, max_features=max_features, labels=labels)
    micro = data["micro_scale"]
    base = data["baseline_micro"] / micro
    final = data["raw_micro"] / micro
    segments = data["segments"]

    n_rows = len(segments) + 2  # base row + segments + score row
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 0.55 * n_rows + 1.2))
    else:
        fig = ax.figure

    y = n_rows - 1
    ax.plot([base, base], [y - 0.4, -0.6], color=colors["base"], lw=0.8, ls=":", zorder=1)
    ax.scatter([base], [y], color=colors["base"], marker="D", zorder=3)
    ax.annotate(f"  base {base:.4f}", (base, y), va="center", fontsize=8)
    ticks, tick_labels = [y], ["Baseline"]

    cursor = base
    for seg in segments:
        y -= 1
        delta = seg["latent_delta"]
        ax.barh(
            y,
            delta,
            left=cursor,
            height=0.62,
            color=_segment_color(seg, colors),
            edgecolor="white",
            zorder=2,
        )
        note = f"{delta:+.4f}"
        if seg["impact_int"] is not None:
            note += f"  ({seg['impact_int']:+d})"
        anchor = cursor + delta
        ax.annotate(
            f"  {note}" if delta >= 0 else f"{note}  ",
            (anchor, y),
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=8,
        )
        cursor += delta
        ticks.append(y)
        tick_labels.append(seg["label"])

    y -= 1
    ax.barh(y, final, height=0.62, color=colors["base"], zorder=2)
    ax.annotate(f"  {final:.4f}", (final, y), va="center", fontsize=8, fontweight="bold")
    ticks.append(y)
    tick_labels.append("Score")

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("latent (probability scale)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title is None:
        title = f"Band {data['band']}  ·  latent_int {data['latent_int']}" + (
            f"  ·  PD {data['pd']:.4f}" if data.get("pd") is not None else ""
        )
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig, ax


def decision_drivers(
    decisions: list[dict],
    *,
    y=None,
    top: int = 15,
    labels: dict | None = None,
    colors: dict | None = None,
    ax=None,
):
    """Population view: exact per-feature impacts across many decisions.

    Strip plot sorted by mean absolute impact; the black tick marks each
    feature's mean impact. Color by outcome when ``y`` is given.

    Returns:
        (figure, axes)
    """
    colors = {**COLORS, **(colors or {})}
    names, rows = driver_table(decisions, labels=labels)
    impacts = np.asarray(rows)
    order = np.argsort(-np.abs(impacts).mean(axis=0))[:top]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.4))
    else:
        fig = ax.figure

    rng = np.random.default_rng(0)  # deterministic jitter
    y_arr = None if y is None else np.asarray(y).reshape(-1)
    for row_pos, j in enumerate(order):
        yy = len(order) - 1 - row_pos + rng.uniform(-0.28, 0.28, size=len(impacts))
        point_colors = (
            colors["neutral"]
            if y_arr is None
            else np.where(y_arr == 1, colors["bad"], colors["good"])
        )
        ax.scatter(impacts[:, j], yy, s=9, alpha=0.45, c=point_colors, linewidths=0, zorder=2)
        ax.scatter(
            impacts[:, j].mean(),
            len(order) - 1 - row_pos,
            marker="|",
            s=380,
            color="#111111",
            zorder=3,
        )

    ax.axvline(0, color="#666666", lw=0.8, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[j] for j in order][::-1], fontsize=9)
    ax.set_xlabel("impact on latent (exact, per decision)")
    ax.set_title(
        "Decision drivers — sorted by mean |impact|", loc="left", fontsize=10, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_arr is not None:
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=colors["good"], label="repaid (y=0)"),
            plt.Line2D([], [], marker="o", ls="", color=colors["bad"], label="default (y=1)"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig, ax


def band_drivers(
    decisions: list[dict],
    *,
    y=None,
    top_k: int = 5,
    cols: int = 3,
    labels: dict | None = None,
    colors: dict | None = None,
):
    """Per-band small multiples of decision drivers.

    Reveals what separates each band's population — the segment-level
    diagnostic. Returns (figure, axes array).
    """
    colors = {**COLORS, **(colors or {})}
    names, rows = driver_table(decisions, labels=labels)
    impacts = np.asarray(rows)
    y_arr = None if y is None else np.asarray(y).reshape(-1)
    bands = sorted({str(d["band"]) for d in decisions})
    band_of = np.array([str(d["band"]) for d in decisions])

    n_rows = math.ceil(len(bands) / cols)
    fig, axes = plt.subplots(
        n_rows, cols, figsize=(3.4 * cols, 2.6 * n_rows), squeeze=False, sharex=True
    )
    rng = np.random.default_rng(0)

    for pos, band in enumerate(bands):
        ax = axes[pos // cols][pos % cols]
        mask = band_of == band
        sub = impacts[mask]
        order = np.argsort(-np.abs(sub).mean(axis=0))[:top_k]
        for row_pos, j in enumerate(order):
            yy = top_k - 1 - row_pos + rng.uniform(-0.25, 0.25, size=mask.sum())
            point_colors = (
                colors["neutral"]
                if y_arr is None
                else np.where(y_arr[mask] == 1, colors["bad"], colors["good"])
            )
            ax.scatter(sub[:, j], yy, s=7, alpha=0.45, c=point_colors, linewidths=0)
        ax.axvline(0, color="#666666", lw=0.7)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(
            [names[j] for j in order][::-1][:top_k] + [""] * max(0, top_k - len(order)),
            fontsize=7,
        )
        ax.set_title(f"Band {band}  (n={int(mask.sum())})", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)

    for pos in range(len(bands), n_rows * cols):
        axes[pos // cols][pos % cols].axis("off")
    fig.suptitle("Decision drivers by band", x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


def band_ladder(decisions: list[dict], y, *, ax=None, colors: dict | None = None):
    """Observed bad rate per band — the monotonicity picture.

    Accepts score-only payloads (``explain=False``); banding needs no
    contributions. Returns (figure, axes).
    """
    colors = {**COLORS, **(colors or {})}
    table = band_table(decisions, y)
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.75 * len(table) + 2.4, 3.4))
    else:
        fig = ax.figure

    bands = [t["band"] for t in table]
    rates = [t["bad_rate"] for t in table]
    ax.bar(bands, rates, color=colors["bad"], alpha=0.85, zorder=2)
    ax.plot(bands, rates, color="#111111", lw=1.1, marker="o", ms=3.5, zorder=3)
    for i, t in enumerate(table):
        ax.annotate(
            f"n={t['n']:,}",
            (i, 0),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#666666",
            annotation_clip=False,
        )
    ax.set_ylabel("observed bad rate")
    ax.set_title("Bad rate by band", loc="left", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax
