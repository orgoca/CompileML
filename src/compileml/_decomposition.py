"""Exact decomposition of a mean score gap into per-feature parts.

One implementation, called by fairness §6 (two groups) and by monitoring
(reference and current populations). Sums stay integers and means are exact
rationals, so the parts add back to the gap with ``==`` on any machine and in
any row order. A second copy is how the float means fixed in #56 would come
back, so there is not one.
"""

from __future__ import annotations

from fractions import Fraction


def require_exact_contributions(
    decisions, caller: str, *, advice: str = "Compile at depth <= 2."
) -> None:
    """Refuse decisions that cannot be decomposed exactly."""
    if not decisions:
        raise ValueError(f"{caller} needs at least one decision on each side")
    if any("contributions" not in d for d in decisions):
        raise ValueError(
            f"{caller} needs per-feature contributions; call "
            "decide(artifact, row, include_contributions=True). The default "
            "payload carries 'attribution' as a method label only."
        )
    residual = max(abs(int(d["attribution_residual_half_micro"])) for d in decisions)
    if residual != 0:
        raise ValueError(
            f"attribution residual is {residual}, not zero: this artifact's "
            "attribution is not exact (whitebox depth > 2), so a decomposition "
            f"would not sum to the gap. {advice}"
        )


def _integer_sums(decisions, n_features: int) -> tuple[int, list[int]]:
    movement = 0
    contributions = [0] * n_features
    for d in decisions:
        movement += 2 * (int(d["raw_micro"]) - int(d["baseline_micro"]))
        for c in d["contributions"]:
            contributions[c["index"]] += int(c["impact_half_micro"])
    return movement, contributions


def mean_gap_by_feature(decisions_a, decisions_b, feature_names) -> dict:
    """Mean raw score movement of side A minus side B, decomposed by feature.

    Movement is ``2 * (raw_micro - baseline_micro)`` in half-micro units. Both
    sides share the artifact's baseline, so the gap is exactly the difference
    in mean raw score. Callers check exactness first with
    :func:`require_exact_contributions`.
    """
    p = len(feature_names)
    n_a, n_b = len(decisions_a), len(decisions_b)
    movement_a, sums_a = _integer_sums(decisions_a, p)
    movement_b, sums_b = _integer_sums(decisions_b, p)

    # Only the division is rational; nothing becomes a float until reported.
    gap = Fraction(movement_a, n_a) - Fraction(movement_b, n_b)
    per = [Fraction(sums_a[j], n_a) - Fraction(sums_b[j], n_b) for j in range(p)]
    total = sum(per, Fraction(0))

    rows = [
        {
            "feature": str(feature_names[j]),
            "gap_half_micro": float(per[j]),
            "share_pct": float(100 * per[j] / total) if total else float("nan"),
        }
        for j in range(p)
    ]
    return {
        "mean_gap_half_micro": float(gap),
        "sum_of_feature_gaps": float(total),
        "residual": float(gap - total),
        "by_feature": sorted(rows, key=lambda r: -abs(r["gap_half_micro"])),
    }
