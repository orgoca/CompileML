"""Exact integer attribution (ARTIFACT_SPEC.md §7).

Contributions are carried in half-micro units so that the Shapley-style
pairwise allocation ``main_effect - interaction_sum / 2`` stays integral.
The reconciliation identity

    2 * (full - fbase)  ==  sum(c2)  +  residual2

holds exactly, in integers, for every artifact; ``residual2 == 0``
whenever the compiled trees have depth <= 2.

Cost note: this computes 1 + n + n*(n-1)/2 ensemble traversals per row
(one per feature, one per feature pair). It is exact, not sampled — but
it is O(n^2) in feature count. See the spec's limits table.
"""

from __future__ import annotations

from collections.abc import Sequence

from compileml.runtime._intmath import div_rha
from compileml.runtime.score import score_micro


def contributions_half_micro(
    model: dict, x: Sequence[float], baseline: Sequence[float]
) -> tuple[list[int], int, int, int]:
    """Per-feature contributions in half-micro units.

    Returns (c2, full, fbase, residual2) where all values are integers,
    c2[j] is twice the micro-unit contribution of feature j, and
    residual2 satisfies the spec §7.4 identity exactly.
    """
    n = len(x)
    row = list(x)
    full = score_micro(model, row)
    fbase = score_micro(model, list(baseline))

    # Main effects: d[j] = full - S(x with j at baseline)
    perturbed = [0] * n
    for j in range(n):
        kept = row[j]
        row[j] = baseline[j]
        perturbed[j] = score_micro(model, row)
        row[j] = kept
    d = [full - perturbed[j] for j in range(n)]

    # Pairwise interactions, accumulated per feature.
    isum = [0] * n
    for i in range(n):
        kept_i = row[i]
        row[i] = baseline[i]
        for j in range(i + 1, n):
            kept_j = row[j]
            row[j] = baseline[j]
            f_both = score_micro(model, row)
            row[j] = kept_j
            inter = full - perturbed[i] - perturbed[j] + f_both
            isum[i] += inter
            isum[j] += inter
        row[i] = kept_i

    c2 = [2 * d[j] - isum[j] for j in range(n)]
    residual2 = 2 * (full - fbase) - sum(c2)
    return c2, full, fbase, residual2


def display_impacts(
    c2: Sequence[int], full: int, fbase: int, residual2: int, ratio: int
) -> list[int]:
    """Display-scale impacts that sum exactly to the display-scale target (§7.5).

    Uses largest-remainder allocation: floor-divide every contribution, then
    hand out the missing units to the largest remainders (ties: lower index).
    """
    ratio2 = 2 * ratio
    target = div_rha(2 * (full - fbase) - residual2, ratio2)

    q = [c // ratio2 for c in c2]  # floor toward -inf
    r = [c2[j] - q[j] * ratio2 for j in range(len(c2))]
    deficit = target - sum(q)

    if deficit > 0:
        order = sorted(range(len(c2)), key=lambda j: (-r[j], j))
        for j in order[:deficit]:
            q[j] += 1
    return q


def format_reasons(
    c2: Sequence[int],
    impact_int: Sequence[int],
    feature_names: Sequence[str],
    reasons_dict: dict,
    display_names: dict,
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    """Adverse and favorable reason blocks (§7.6).

    Positive contribution = risk-increasing = adverse. Suppressed features
    are excluded from reasons (their contribution still exists).
    """

    def suppressed(name: str) -> bool:
        entry = reasons_dict.get(name)
        return bool(entry and entry.get("suppress"))

    eligible = [j for j in range(len(c2)) if c2[j] != 0 and not suppressed(feature_names[j])]
    adverse = sorted((j for j in eligible if c2[j] > 0), key=lambda j: (-c2[j], j))[:top_k]
    favorable = sorted((j for j in eligible if c2[j] < 0), key=lambda j: (c2[j], j))[:top_k]

    def block(j: int, is_adverse: bool) -> dict:
        name = feature_names[j]
        entry = reasons_dict.get(name, {})
        if is_adverse:
            code = str(entry.get("code", f"NEGATIVE_{name}"))
            message = str(entry.get("negative", f"{name} increased the estimated risk."))
        else:
            code = str(entry.get("code", f"POSITIVE_{name}"))
            message = str(entry.get("positive", f"{name} reduced the estimated risk."))
        return {
            "code": code,
            "feature": name,
            "label": str(display_names.get(name, name)),
            "impact_half_micro": int(c2[j]),
            "impact_int": int(impact_int[j]),
            "direction": "risk_increasing" if is_adverse else "risk_decreasing",
            "message": message,
        }

    return (
        [block(j, True) for j in adverse],
        [block(j, False) for j in favorable],
    )
