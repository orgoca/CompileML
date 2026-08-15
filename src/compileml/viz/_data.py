"""Payload → plot-ready data. Standard library only.

The design rule of ``compileml.viz``: **plots draw decision payloads, they
never recompute them.** Every function here consumes the output of
``decide(..., include_contributions=True)`` — the same integers production
emits — so a chart can never disagree with the deployed decision. The
waterfall's bars sum to the score because the spec §7.4 reconciliation
identity says they must, and that invariant is asserted in tests on the
exact half-micro integers.
"""

from __future__ import annotations


def _require_contributions(decision: dict) -> list[dict]:
    contributions = decision.get("contributions")
    if not contributions:
        raise ValueError(
            "decision payload has no contributions — call "
            "decide(artifact, row, include_contributions=True)"
        )
    return contributions


def waterfall_segments(
    decision: dict, *, max_features: int = 10, labels: dict | None = None
) -> dict:
    """Exact waterfall segments for one decision.

    Returns a dict with ``baseline_micro``, ``raw_micro``, ``micro_scale``,
    and ``segments`` — each segment carrying its exact ``half_micro`` integer
    and its float ``latent_delta``. Invariant (exact, integer arithmetic):

        2 * (raw_micro - baseline_micro) == sum(seg.half_micro) over segments
    """
    contributions = _require_contributions(decision)
    labels = labels or {}
    micro_scale = int(decision["micro_scale"])
    baseline_micro = int(decision["baseline_micro"])
    raw_micro = int(decision["raw_micro"])
    residual2 = int(decision.get("attribution_residual_half_micro", 0))

    ordered = sorted(contributions, key=lambda c: (-abs(int(c["impact_half_micro"])), c["index"]))
    shown = [c for c in ordered[:max_features] if c["impact_half_micro"] != 0]
    rest_half = sum(int(c["impact_half_micro"]) for c in ordered[max_features:])

    segments = [
        {
            "label": str(labels.get(c["feature"], c["feature"])),
            "half_micro": int(c["impact_half_micro"]),
            "latent_delta": int(c["impact_half_micro"]) / (2 * micro_scale),
            "impact_int": int(c["impact_int"]),
            "kind": "impact",
        }
        for c in shown
    ]
    if rest_half:
        segments.append(
            {
                "label": f"{len(ordered) - len(shown)} smaller features",
                "half_micro": rest_half,
                "latent_delta": rest_half / (2 * micro_scale),
                "impact_int": None,
                "kind": "remainder",
            }
        )
    if residual2:
        segments.append(
            {
                "label": "interaction residual (depth > 2)",
                "half_micro": residual2,
                "latent_delta": residual2 / (2 * micro_scale),
                "impact_int": None,
                "kind": "residual",
            }
        )
    # Zero-impact features dropped from `shown` still balance the identity:
    # they contribute exactly 0 half-micro units by construction.
    return {
        "baseline_micro": baseline_micro,
        "raw_micro": raw_micro,
        "micro_scale": micro_scale,
        "band": decision.get("band"),
        "latent_int": decision.get("latent_int"),
        "pd": decision.get("pd"),
        "segments": segments,
    }


def driver_table(
    decisions: list[dict], *, labels: dict | None = None
) -> tuple[list[str], list[list[float]]]:
    """(feature_names, impacts) across many decisions, in latent units.

    ``impacts[i][j]`` is decision *i*'s exact contribution for feature *j*.
    Feature order follows the artifact's feature order.
    """
    if not decisions:
        raise ValueError("no decisions given")
    labels = labels or {}
    first = _require_contributions(decisions[0])
    names = [
        str(labels.get(c["feature"], c["feature"])) for c in sorted(first, key=lambda c: c["index"])
    ]

    rows: list[list[float]] = []
    for decision in decisions:
        contributions = sorted(_require_contributions(decision), key=lambda c: c["index"])
        micro_scale = int(decision["micro_scale"])
        rows.append([int(c["impact_half_micro"]) / (2 * micro_scale) for c in contributions])
    return names, rows


def band_table(decisions: list[dict], y=None) -> list[dict]:
    """Per-band counts (and bad rates when outcomes are given).

    Works with score-only payloads (``explain=False``) — banding needs no
    contributions.
    """
    if y is not None and len(y) != len(decisions):
        raise ValueError("y must align with decisions")
    stats: dict[str, dict] = {}
    for i, decision in enumerate(decisions):
        band = str(decision["band"])
        entry = stats.setdefault(band, {"band": band, "n": 0, "bad": 0})
        entry["n"] += 1
        if y is not None:
            entry["bad"] += int(y[i])
    table = [stats[k] for k in sorted(stats)]
    for entry in table:
        entry["bad_rate"] = (entry["bad"] / entry["n"]) if (y is not None and entry["n"]) else None
    return table
