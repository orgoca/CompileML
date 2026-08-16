"""Scorecard renderers: markdown for humans, CSV for spreadsheets.

Points are shown in micro units (the artifact's exact integers) with a
display-scale approximation alongside — the micro column is the one that
re-sums to the decision.
"""

from __future__ import annotations

import csv
import io


def _pts(points_micro: int, micro_scale: int, scale: int) -> str:
    ratio = micro_scale // scale
    return f"{points_micro:+,} µ ({points_micro / ratio:+.1f} pts)"


def scorecard_to_markdown(scorecard: dict, *, labels: dict | None = None) -> str:
    """Render the scorecard as markdown tables."""
    labels = labels or {}
    micro, scale = scorecard["micro_scale"], scorecard["scale"]
    out: list[str] = []
    push = out.append

    push("# Scorecard")
    push("")
    push(f"- artifact: `{(scorecard.get('artifact_hash') or 'unknown')[:16]}…`")
    push(f"- trees: {scorecard['n_trees']} · depth: {scorecard['whitebox_max_depth']} · exact: yes")
    push(f"- base points: **{_pts(scorecard['base_points_micro'], micro, scale)}**")
    push("")
    push("Points are the artifact's integers (micro units); base + bins + grids")
    push("re-sum to every decision's `raw_micro` exactly.")

    for name, effect in scorecard["main_effects"].items():
        push("")
        push(f"## {labels.get(name, name)}")
        push("")
        push("| Bin | Points |")
        push("|---|---:|")
        for b in effect["bins"]:
            push(f"| {b['interval']} | {_pts(b['points_micro'], micro, scale)} |")

    for inter in scorecard["interactions"].values():
        row_name = labels.get(inter["row_feature"], inter["row_feature"])
        col_name = labels.get(inter["col_feature"], inter["col_feature"])
        push("")
        push(f"## Interaction: {row_name} × {col_name}")
        push("")
        push(
            "| " + row_name + " \\ " + col_name + " | " + " | ".join(inter["col_intervals"]) + " |"
        )
        push("|---|" + "---:|" * len(inter["col_intervals"]))
        for r_label, row in zip(inter["row_intervals"], inter["grid_micro"]):
            cells = " | ".join(_pts(v, micro, scale) for v in row)
            push(f"| {r_label} | {cells} |")

    push("")
    return "\n".join(out)


def scorecard_to_csv(scorecard: dict) -> str:
    """Flat CSV: one row per bin / grid cell, exact micro points."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["kind", "feature", "interval", "feature_2", "interval_2", "points_micro"])
    writer.writerow(["base", "", "", "", "", scorecard["base_points_micro"]])
    for name, effect in scorecard["main_effects"].items():
        for b in effect["bins"]:
            writer.writerow(["main", name, b["interval"], "", "", b["points_micro"]])
    for inter in scorecard["interactions"].values():
        for r_label, row in zip(inter["row_intervals"], inter["grid_micro"]):
            for c_label, v in zip(inter["col_intervals"], row):
                writer.writerow(
                    ["interaction", inter["row_feature"], r_label, inter["col_feature"], c_label, v]
                )
    return buf.getvalue()
