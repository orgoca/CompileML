"""Matplotlib renderings of decision payloads (``compileml[viz]``).

Every function takes ``decide()`` output — the plots draw the deployed
integers, they never recompute them. See ``_data.py`` for the exact
segment/table construction these renderers share with the SVG exporter.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from compileml.viz._arrow import DEAD_ZONE, HALF_H, OVERHANG, TICK_HALF, arrow_points, tri_frac
from compileml.viz._data import band_table, waterfall_segments

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

    # Fix the view first: the arrow-head fraction is computed in PIXELS
    # (data-space widths are the classic porting bug), so the data→pixel
    # mapping must be final before any polygon is built.
    cursor, walk = base, [base, final]
    for seg in segments:
        cursor += seg["latent_delta"]
        walk.append(cursor)
    lo_data, hi_data = min(walk), max(walk)
    span = (hi_data - lo_data) or 1.0
    ax.set_xlim(min(lo_data - 0.10 * span, 0.0), hi_data + 0.16 * span)
    ax.set_ylim(-0.6, n_rows - 0.4)

    y = n_rows - 1
    ax.plot([base, base], [y - 0.4, -0.6], color=colors["base"], lw=0.8, ls=":", zorder=1)
    ax.scatter([base], [y], color=colors["base"], marker="D", zorder=3)
    ax.annotate(f"  base {base:.4f}", (base, y), va="center", fontsize=8)
    ticks, tick_labels = [y], ["Baseline"]

    cursor = base
    rows = []
    for seg in segments:
        y -= 1
        rows.append((y, seg, cursor))
        note = f"{seg['latent_delta']:+.4f}"
        if seg["impact_int"] is not None:
            note += f"  ({seg['impact_int']:+d})"
        anchor = cursor + seg["latent_delta"]
        ax.annotate(
            f"  {note}" if seg["latent_delta"] >= 0 else f"{note}  ",
            (anchor, y),
            va="center",
            ha="left" if seg["latent_delta"] >= 0 else "right",
            fontsize=8,
        )
        cursor += seg["latent_delta"]
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

    # Realize the layout, then build the arrows with true pixel geometry.
    fig.tight_layout()
    fig.canvas.draw()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    px_per_x = ax.bbox.width / (xlim[1] - xlim[0])
    px_per_y = ax.bbox.height / (ylim[1] - ylim[0])
    scale_px = (hi_data - lo_data) * px_per_x  # pixel width of the data walk
    half_h = HALF_H / px_per_y
    overhang = OVERHANG / px_per_y

    for row_y, seg, x_start in rows:
        delta = seg["latent_delta"]
        abs_px = abs(delta) * px_per_x
        color = _segment_color(seg, colors)
        if abs_px < DEAD_ZONE:
            tick_half = TICK_HALF / px_per_y
            ax.plot(
                [x_start, x_start],
                [row_y - tick_half, row_y + tick_half],
                color=color,
                lw=1.5,
                zorder=2,
            )
            continue
        tri_len = abs(delta) * tri_frac(abs_px, scale_px)
        points = arrow_points(
            x_start,
            x_start + delta,
            row_y,
            half_h=half_h,
            overhang=overhang,
            tri_len=tri_len,
            dead_zone=0.0,  # pixel dead-zone already checked above (abs_px)
        )
        ax.add_patch(
            Polygon(points, closed=True, facecolor=color, alpha=0.92, edgecolor="none", zorder=2)
        )
    return fig, ax


# Dark low-end counterparts for the value-gradient encodings — original design.
COLORS_LOW = {"good": "#1a5c35", "bad": "#7d3c1a", "neutral": "#424949"}

X_LABEL = "Delta Latent (Whitebox)"


def _resolve_color_by(color_by: str, has_y: bool) -> str:
    valid = {"auto", "impact", "outcome"}
    if color_by not in valid:
        raise ValueError("color_by must be one of: auto, impact, outcome")
    if color_by == "auto":
        return "outcome" if has_y else "impact"
    if color_by == "outcome" and not has_y:
        return "neutral"
    return color_by


def _reason_rows(decisions: list[dict], y=None, values=None) -> dict:
    """Flatten payload reason blocks into plot arrays (code, impact, y, band).

    Mirrors the original reason-row semantics: one point per (decision,
    top-k reason), labeled by reason CODE, impact in latent units.
    """
    codes, impacts, outcomes, bands, feats, vals = [], [], [], [], [], []
    for i, decision in enumerate(decisions):
        if "reasons_negative" not in decision:
            raise ValueError(
                "decision payloads carry no reason blocks — call decide(..., explain=True)"
            )
        # A decision whose every reason is zero-impact or suppressed simply
        # contributes no points — same as the original reason-row builder.
        blocks = decision["reasons_negative"] + decision["reasons_positive"]
        micro = int(decision["micro_scale"])
        for block in blocks:
            codes.append(str(block["code"]))
            impacts.append(int(block["impact_half_micro"]) / (2 * micro))
            outcomes.append(None if y is None else int(y[i]))
            bands.append(str(decision.get("band")))
            feats.append(str(block.get("feature", "")))
            vals.append(None if values is None else values[i].get(block.get("feature")))
    return {
        "code": np.array(codes, dtype=object),
        "impact": np.array(impacts, dtype=float),
        "y": None if y is None else np.array(outcomes, dtype=float),
        "band": np.array(bands, dtype=object),
        "feature": np.array(feats, dtype=object),
        "value": np.array(vals, dtype=object) if values is not None else None,
    }


def _pile_jitter(codes, impacts, rng, base=0.06, maxw=0.18):
    """Tiny x-jitter to break vertical piles caused by discrete point values.

    Applied only within each exact (code, impact) pile — original design.
    """
    x = np.asarray(impacts, dtype=float)
    x_jit = np.zeros_like(x)
    piles: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(codes, x)):
        piles.setdefault(key, []).append(i)
    for members in piles.values():
        m = len(members)
        if m <= 1:
            continue
        w = min(maxw, base * np.sqrt(m))
        x_jit[members] = rng.uniform(-w, w, size=m)
    return x + x_jit


def _pct_rank_per_feature(features, raw_values) -> np.ndarray:
    """Percentile-rank values within each feature (average ties, NaN→0.5)."""
    out = np.full(len(features), 0.5)
    values = np.array([np.nan if v is None else float(v) for v in raw_values], dtype=float)
    for feat in set(features):
        idx = np.where(features == feat)[0]
        vals = values[idx]
        ok = ~np.isnan(vals)
        if ok.sum() < 2:
            continue
        order = vals[ok].argsort().argsort().astype(float)  # ranks 0..n-1
        out[idx[ok]] = (order + 0.5) / ok.sum()
    return out


def _scatter_groups(
    ax,
    x_plot,
    y_plot,
    rows,
    mode,
    colors,
    *,
    point_size,
    alpha_good,
    alpha_bad,
    value_color,
    value_alpha,
    alpha_range,
    colors_low,
):
    """The original scatter geometry: good/bad/neutral masks, bad points one
    size larger, more opaque, and drawn on top; optional per-point value
    encodings (color gradient toward a dark low-end, or opacity ramp)."""
    from matplotlib.colors import to_rgb

    rank = None
    if (value_color or value_alpha) and rows["value"] is not None:
        rank = _pct_rank_per_feature(rows["feature"], rows["value"])
    use_color = value_color and rank is not None
    use_alpha = value_alpha and not use_color and rank is not None

    def paint(mask, key, size, alpha, zorder):
        if not mask.any():
            return
        if use_color:
            lo = np.array(to_rgb(colors_low[key]))
            hi = np.array(to_rgb(colors[key]))
            t = np.clip(rank[mask], 0.0, 1.0)[:, None]
            c_arr = np.hstack([lo + t * (hi - lo), np.full((mask.sum(), 1), alpha)])
            ax.scatter(
                x_plot[mask], y_plot[mask], c=c_arr, s=size, edgecolors="none", zorder=zorder
            )
        elif use_alpha:
            a_min, a_max = float(alpha_range[0]), float(alpha_range[1])
            rgb = np.array(to_rgb(colors[key]))
            c_arr = np.column_stack(
                [np.tile(rgb, (mask.sum(), 1)), a_min + rank[mask] * (a_max - a_min)]
            )
            ax.scatter(
                x_plot[mask], y_plot[mask], c=c_arr, s=size, edgecolors="none", zorder=zorder
            )
        else:
            ax.scatter(
                x_plot[mask],
                y_plot[mask],
                c=colors[key],
                s=size,
                alpha=alpha,
                edgecolors="none",
                zorder=zorder,
            )

    if mode == "impact":
        impact = rows["impact"]
        paint(impact < 0, "good", point_size, alpha_good, 1)
        paint(impact > 0, "bad", point_size + 1, alpha_bad, 2)
        paint(impact == 0, "neutral", point_size, 0.7, 1)
    elif mode == "outcome":
        y_arr = rows["y"]
        paint(y_arr == 0, "good", point_size, alpha_good, 1)
        paint(y_arr == 1, "bad", point_size + 1, alpha_bad, 2)
    else:
        paint(np.ones(len(x_plot), dtype=bool), "neutral", point_size, 0.7, 1)


def decision_drivers(
    decisions: list[dict],
    *,
    y=None,
    values=None,
    sort_metric: str = "mean",
    top_codes: int = 20,
    point_size: int = 6,
    alpha_good: float = 0.65,
    alpha_bad: float = 0.75,
    y_jitter: float = 0.34,
    xj_base: float = 0.06,
    xj_max: float = 0.18,
    seed: int = 0,
    colors: dict | None = None,
    colors_low: dict | None = None,
    color_by: str = "auto",
    value_color: bool = False,
    value_alpha: bool = False,
    alpha_range: tuple = (0.20, 0.90),
    ax=None,
):
    """Global decision drivers (deterministic SHAP-style beeswarm).

    Original design, payload-driven: one point per (decision, top-k reason),
    labeled by reason code, biggest drivers on top. ``color_by``: "auto"
    (outcome when ``y`` given, else impact direction), "impact", or
    "outcome". ``value_color`` / ``value_alpha`` encode per-point feature
    values (pass ``values`` as one {feature: value} dict per decision).

    Returns:
        (figure, axes)
    """
    colors = {**COLORS, **(colors or {})}
    colors_low = {**COLORS_LOW, **(colors_low or {})}
    rows = _reason_rows(decisions, y=y, values=values)

    metric_fns = {
        "mean": lambda s: float(np.mean(np.abs(s))),
        "sum": lambda s: float(np.sum(np.abs(s))),
        "max": lambda s: float(np.max(np.abs(s))),
    }
    if sort_metric not in metric_fns:
        raise ValueError("sort_metric must be one of: mean, sum, max")

    metric = metric_fns[sort_metric]
    all_codes = list(dict.fromkeys(rows["code"]))
    scored = sorted(all_codes, key=lambda c: metric(rows["impact"][rows["code"] == c]))[-top_codes:]
    keep = np.isin(rows["code"], scored)
    rows = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in rows.items()}

    y_map = {c: j for j, c in enumerate(scored)}  # ascending: biggest on top
    rng = np.random.default_rng(seed)
    y_base = np.array([y_map[c] for c in rows["code"]], dtype=float)
    y_plot = y_base + rng.uniform(-y_jitter, y_jitter, size=len(y_base))
    x_plot = _pile_jitter(rows["code"], rows["impact"], rng, base=xj_base, maxw=xj_max)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    else:
        fig = ax.figure

    has_y = rows["y"] is not None
    mode = _resolve_color_by(color_by, has_y)
    _scatter_groups(
        ax,
        x_plot,
        y_plot,
        rows,
        mode,
        colors,
        point_size=point_size,
        alpha_good=alpha_good,
        alpha_bad=alpha_bad,
        value_color=value_color,
        value_alpha=value_alpha,
        alpha_range=alpha_range,
        colors_low=colors_low,
    )

    means = [float(np.mean(rows["impact"][rows["code"] == c])) for c in scored]
    ax.scatter(
        means, list(range(len(scored))), color="black", marker="|", s=140, linewidth=1.8, zorder=3
    )

    ax.set_yticks(range(len(scored)))
    ax.set_yticklabels(scored, fontsize=10)
    ax.axvline(0.0, color="black", linewidth=1.2, alpha=0.35, linestyle="--")
    ax.set_title(
        f"Decision Drivers (Sorted by {sort_metric.title()} Abs Impact)",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(X_LABEL)
    ax.grid(True, axis="x", alpha=0.15)

    legend_elements = []
    if mode == "impact":
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Risk-decreasing impact",
                    markerfacecolor=colors["good"],
                    markersize=8,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Risk-increasing impact",
                    markerfacecolor=colors["bad"],
                    markersize=8,
                ),
            ]
        )
    elif mode == "outcome":
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Repayment (y=0)",
                    markerfacecolor=colors["good"],
                    markersize=8,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Default (y=1)",
                    markerfacecolor=colors["bad"],
                    markersize=8,
                ),
            ]
        )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="|",
            color="black",
            label="Mean Impact",
            markersize=10,
            linestyle="None",
        )
    )
    if value_color and rows["value"] is not None:
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Color: low feature value",
                    markerfacecolor=colors_low["bad"],
                    markersize=8,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Color: high feature value",
                    markerfacecolor=colors["bad"],
                    markersize=8,
                ),
            ]
        )
    elif value_alpha and rows["value"] is not None:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Opacity = feature value (low → high)",
                markerfacecolor="gray",
                markersize=8,
                alpha=0.55,
            )
        )
    ax.legend(handles=legend_elements, loc="upper right")
    fig.tight_layout()
    return fig, ax


def band_conditioned_decision_drivers(
    decisions: list[dict],
    *,
    y=None,
    target_band: str = "all",
    cols: int = 3,
    top_codes: int = 10,
    point_size: int = 6,
    alpha_good: float = 0.65,
    alpha_bad: float = 0.75,
    y_jitter: float = 0.34,
    xj_base: float = 0.06,
    xj_max: float = 0.18,
    seed: int = 0,
    colors: dict | None = None,
    color_by: str = "auto",
):
    """Band-conditioned decision drivers (faceted beeswarm by band).

    Original design, payload-driven: one panel per band, each ranking its
    own top reason codes by mean absolute impact. Returns (figure, axes).
    """
    colors = {**COLORS, **(colors or {})}
    rows = _reason_rows(decisions, y=y)

    all_bands = sorted(set(rows["band"]))
    if target_band != "all":
        if target_band not in all_bands:
            raise ValueError(f"Band {target_band} not found.")
        display_bands = [target_band]
    else:
        display_bands = all_bands

    n_plots = len(display_bands)
    nrows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=cols, figsize=(6 * cols, 4.6 * nrows), dpi=140, sharex=True
    )
    axes = np.array(axes).reshape(-1)
    rng = np.random.default_rng(seed)

    last_i = -1
    for i, band in enumerate(display_bands):
        last_i = i
        ax = axes[i]
        mask = rows["band"] == band
        if not mask.any():
            ax.text(0.5, 0.5, f"Band {band}\nNo Data", ha="center")
            ax.axis("off")
            continue
        sub = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in rows.items()}

        codes = list(dict.fromkeys(sub["code"]))
        code_order = sorted(
            codes, key=lambda c: float(np.mean(np.abs(sub["impact"][sub["code"] == c])))
        )[-top_codes:]
        keep = np.isin(sub["code"], code_order)
        sub = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in sub.items()}

        y_map = {c: j for j, c in enumerate(code_order)}
        y_base = np.array([y_map[c] for c in sub["code"]], dtype=float)
        y_plot = y_base + rng.uniform(-y_jitter, y_jitter, size=len(y_base))
        x_plot = _pile_jitter(sub["code"], sub["impact"], rng, base=xj_base, maxw=xj_max)

        mode = _resolve_color_by(color_by, sub["y"] is not None)
        _scatter_groups(
            ax,
            x_plot,
            y_plot,
            sub,
            mode,
            colors,
            point_size=point_size,
            alpha_good=alpha_good,
            alpha_bad=alpha_bad,
            value_color=False,
            value_alpha=False,
            alpha_range=(0.20, 0.90),
            colors_low=COLORS_LOW,
        )

        means = [float(np.mean(sub["impact"][sub["code"] == c])) for c in code_order]
        ax.scatter(
            means,
            list(range(len(code_order))),
            color="black",
            marker="|",
            s=120,
            linewidth=1.6,
            zorder=3,
        )

        ax.axvline(0, color="black", lw=1, alpha=0.35, linestyle="--")
        ax.set_yticks(range(len(code_order)))
        ax.set_yticklabels(code_order, fontsize=8)
        ax.set_title(f"Band {band}", fontweight="bold", loc="left", fontsize=11)
        ax.grid(True, axis="x", alpha=0.12)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for j in range(last_i + 1, len(axes)):
        axes[j].axis("off")

    fig.supxlabel(X_LABEL, fontsize=12, fontweight="bold")

    mode = _resolve_color_by(color_by, rows["y"] is not None)
    legend_elements = []
    if mode == "impact":
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Risk-decreasing impact",
                    markerfacecolor=colors["good"],
                    markersize=8,
                    alpha=0.7,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Risk-increasing impact",
                    markerfacecolor=colors["bad"],
                    markersize=8,
                    alpha=1.0,
                ),
            ]
        )
    elif mode == "outcome":
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Observed repayment (y=0)",
                    markerfacecolor=colors["good"],
                    markersize=8,
                    alpha=0.7,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Observed default (y=1)",
                    markerfacecolor=colors["bad"],
                    markersize=8,
                    alpha=1.0,
                ),
            ]
        )
    else:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Impact points",
                markerfacecolor=colors["neutral"],
                markersize=8,
                alpha=0.7,
            )
        )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="|",
            color="black",
            label="Mean Impact",
            markersize=10,
            linestyle="None",
        )
    )
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )

    # Method wording updated for CompileML's attribution (spec §7): the
    # original said "baseline-substitution"; these payloads carry the exact
    # pairwise-interaction allocation.
    fig.text(
        0.99,
        0.012,
        "Points reflect deterministic pairwise-interaction attribution in latent space.",
        ha="right",
        va="bottom",
        fontsize=10,
        alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
    )

    fig.suptitle(
        "Decision Drivers - Band Conditioned",
        x=0.01,
        ha="left",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.92))
    return fig, axes


# Short alias kept from the first viz release.
band_drivers = band_conditioned_decision_drivers


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
