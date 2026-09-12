"""Exact integer attribution (ARTIFACT_SPEC.md §7).

Contributions are carried in half-micro units so that the Shapley-style
pairwise allocation ``main_effect - interaction_sum / 2`` stays integral.
The reconciliation identity

    2 * (full - fbase)  ==  sum(c2)  +  residual2

holds exactly, in integers, for every artifact; ``residual2 == 0``
whenever the compiled trees have depth <= 2.

Cost: attribution is additive over trees, and a tree responds only to
the features it splits on. Aggregating per tree rather than per
perturbation makes the cost ``O(trees)`` and **independent of feature
count**, instead of the ``2 + n + n(n-1)/2`` ensemble traversals a
perturbation-based derivation needs.

The quantities are unchanged. Integer addition is associative, so
regrouping the same sums is bit-identical rather than merely close —
which is what allows the committed determinism oracle to police this.

The perturbation derivation is kept as
:func:`contributions_half_micro_reference`, and the validation framework
cross-checks one against the other. Two independent derivations agreeing
on every integer is a stronger audit story than one.
"""

from __future__ import annotations

from collections.abc import Sequence

from compileml.runtime._intmath import div_rha
from compileml.runtime.score import LEAF, score_micro

# Per tree the fast path enumerates every subset of the features that tree
# splits on, which is 2**k leaf lookups. At depth <= 2 a tree touches at
# most three features, so k is tiny and the win is large. A deep tree can
# touch 2**depth - 1 features, where the enumeration would explode — those
# fall back to the perturbation derivation, which is linear in trees and
# quadratic in features rather than exponential in either.
MAX_ENUMERATED_TREE_FEATURES = 4


def _tree_leaf(tree: dict, x: Sequence[float]) -> int:
    """One tree's leaf payload for a row. The unit the fast path counts."""
    feature = tree["feature"]
    threshold = tree["threshold"]
    left = tree["left"]
    right = tree["right"]
    node = 0
    while feature[node] != LEAF:
        node = left[node] if x[feature[node]] <= threshold[node] else right[node]
    return tree["value_micro"][node]


def _tree_split_features(tree: dict) -> list[int]:
    return sorted({f for f in tree["feature"] if f != LEAF})


def contributions_half_micro(
    model: dict, x: Sequence[float], baseline: Sequence[float]
) -> tuple[list[int], int, int, int]:
    """Per-feature contributions in half-micro units.

    Returns (c2, full, fbase, residual2) where all values are integers,
    c2[j] is twice the micro-unit contribution of feature j, and
    residual2 satisfies the spec §7.4 identity exactly.

    Computed per tree. A tree that does not split on feature ``j``
    contributes nothing to ``j``'s main effect, because baselining ``j``
    cannot change which leaf the row reaches. The same cancellation is
    sharper for pairs: the interaction term
    ``t(x) - t(x_i) - t(x_j) + t(x_ij)`` vanishes unless the tree splits on
    **both** ``i`` and ``j``. So each tree only needs its own features
    enumerated, and the total is independent of how many features the model
    has.
    """
    trees = model["trees"]
    if any(
        len(_tree_split_features(t)) > MAX_ENUMERATED_TREE_FEATURES for t in trees
    ):  # pragma: no cover - deep trees are out of contract for exactness anyway
        return contributions_half_micro_reference(model, x, baseline)

    n = len(x)
    d = [0] * n
    isum = [0] * n
    full = fbase = int(model["base_micro"])
    row = list(x)

    for tree in trees:
        feats = _tree_split_features(tree)
        k = len(feats)
        if k == 0:  # a stump with no split contributes a constant to both
            constant = _tree_leaf(tree, row)
            full += constant
            fbase += constant
            continue

        # Evaluate this tree with every subset of its own features held at
        # baseline. Bit b of the mask means "feature feats[b] is baselined".
        values = [0] * (1 << k)
        for mask in range(1 << k):
            for b, j in enumerate(feats):
                row[j] = baseline[j] if (mask >> b) & 1 else x[j]
            values[mask] = _tree_leaf(tree, row)
        for j in feats:
            row[j] = x[j]

        at_row = values[0]
        # Baselining every feature the tree reads is indistinguishable from
        # baselining the whole row, since the rest cannot steer it.
        at_baseline = values[(1 << k) - 1]
        full += at_row
        fbase += at_baseline

        for b, j in enumerate(feats):
            d[j] += at_row - values[1 << b]
        for a in range(k):
            for b in range(a + 1, k):
                ma, mb = 1 << a, 1 << b
                interaction = at_row - values[ma] - values[mb] + values[ma | mb]
                isum[feats[a]] += interaction
                isum[feats[b]] += interaction

    c2 = [2 * d[j] - isum[j] for j in range(n)]
    residual2 = 2 * (full - fbase) - sum(c2)
    return c2, full, fbase, residual2


def contributions_half_micro_reference(
    model: dict, x: Sequence[float], baseline: Sequence[float]
) -> tuple[list[int], int, int, int]:
    """The perturbation derivation, kept as an independent cross-check.

    Costs ``2 + n + n(n-1)/2`` ensemble traversals — the row, the baseline,
    each feature perturbed, each pair perturbed — and reaches the same
    integers by a different route. Retained deliberately: the validation
    framework runs both and compares, and two derivations agreeing beats
    one derivation asserting.
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
