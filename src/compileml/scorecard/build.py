"""Exact scorecard extraction from a compiled artifact. Standard library only.

At whitebox depth ≤ 2, a CompileML artifact *is* a classic points-based
scorecard — exactly, not approximately:

- every tree that references **one** feature is a step function over that
  feature; summing the step functions per feature yields the familiar
  ``bin → points`` table;
- every depth-2 tree that references **two** features contributes to an
  explicit pairwise interaction grid over the union of their thresholds;
- root-leaf trees fold into the base points.

Because leaf values are the artifact's integers, the collapse is pure
integer addition: for any input row,

    base_points + Σ main_effect(x) + Σ interaction(x)  ==  raw_micro

bit for bit — asserted by :func:`score_from_scorecard` and the test suite.
Above depth 2 no clean scorecard exists (the same reason attribution stops
being exact), and :func:`build_scorecard` refuses rather than approximates.
"""

from __future__ import annotations

import struct
from bisect import bisect_left

LEAF = -2


def _tree_features(tree: dict) -> list[int]:
    return sorted({f for f in tree["feature"] if f != LEAF})


def _subtree_nodes(tree: dict, start: int) -> list[int]:
    nodes, stack = [], [start]
    while stack:
        node = stack.pop()
        nodes.append(node)
        if tree["feature"][node] != LEAF:
            stack.append(tree["left"][node])
            stack.append(tree["right"][node])
    return nodes


# A *component* is a ≤2-feature piece of the ensemble: either a whole tree
# ("tree") or one root-side of a tree ("half"). The half case exists because
# a depth-2 tree may reference THREE features — root f, left child on g,
# right child on h — which is still only pairwise structure (each leaf is a
# product of at most two indicators), but spans two pairs: (f,g) and (f,h).
# Splitting at the root yields two components of at most two features each,
# and their sum reproduces the tree exactly: the inactive side contributes 0.


def _components(tree: dict):
    feats = _tree_features(tree)
    if len(feats) <= 2:
        yield ("tree", tree, None)
        return
    for side in ("left", "right"):
        yield ("half", tree, side)


def _component_features(kind: str, tree: dict, side) -> list[int]:
    if kind == "tree":
        return _tree_features(tree)
    child = tree[side][0]
    feats = {tree["feature"][0]}
    feats.update(f for n in _subtree_nodes(tree, child) if (f := tree["feature"][n]) != LEAF)
    return sorted(feats)


def _eval_component(kind: str, tree: dict, side, values: dict[int, float]) -> int:
    if kind == "half":
        root_f, root_t = tree["feature"][0], tree["threshold"][0]
        goes_left = values[root_f] <= root_t
        if (side == "left") != goes_left:
            return 0
        node = tree[side][0]
    else:
        node = 0
    while tree["feature"][node] != LEAF:
        f = tree["feature"][node]
        node = tree["left"][node] if values[f] <= tree["threshold"][node] else tree["right"][node]
    return int(tree["value_micro"][node])


def _component_thresholds(kind: str, tree: dict, side, feature: int) -> list[float]:
    if kind == "tree":
        nodes = range(len(tree["feature"]))
    else:
        nodes = [0, *_subtree_nodes(tree, tree[side][0])]
    return [float(tree["threshold"][n]) for n in nodes if tree["feature"][n] == feature]


def _representatives(thresholds: list[float]) -> list[float]:
    """One probe value per interval of the partition induced by thresholds.

    Intervals are ``(-inf, t1], (t1, t2], …, (tk, inf)`` — matching the
    ``x <= t`` split convention. Any value inside an interval resolves every
    comparison identically, so one probe per interval evaluates the trees
    exactly on that interval.
    """
    if not thresholds:
        return [0.0]
    # (-inf, t1]: t1 itself is inside (x == t goes left under `<=`).
    # (a, b]: b is inside (b > a and b <= b). (tk, inf): anything above tk.
    return [thresholds[0], *thresholds[1:], thresholds[-1] + 1.0]


def _interval_label(thresholds: list[float], i: int, decimals: int = 6) -> str:
    def fmt(v: float) -> str:
        return f"{v:.{decimals}g}"

    if not thresholds:
        return "(any value)"
    if i == 0:
        return f"<= {fmt(thresholds[0])}"
    if i == len(thresholds):
        return f"> {fmt(thresholds[-1])}"
    return f"({fmt(thresholds[i - 1])}, {fmt(thresholds[i])}]"


def build_scorecard(artifact: dict) -> dict:
    """Collapse a depth ≤ 2 artifact into an exact points scorecard.

    Returns a dict with ``base_points_micro``, per-feature ``main_effects``
    (interval → integer points), pairwise ``interactions`` (threshold grids
    of integer points), and metadata. All points are in micro units — the
    artifact's own integers — so they re-sum to ``raw_micro`` exactly.
    """
    depth = int(artifact.get("runtime", {}).get("whitebox_max_depth", 99))
    if depth > 2:
        raise ValueError(
            f"artifact records whitebox_max_depth={depth}: no exact scorecard exists "
            "above depth 2 (three-way structure cannot collapse into bins and pairwise "
            "grids). Re-distill with max_depth<=2 — see docs/howto/tuning.md."
        )

    model = artifact["model"]
    names = [str(n) for n in artifact["features"]["names"]]

    base_points = int(model["base_micro"])
    single: dict[int, list[tuple]] = {}
    pairs: dict[tuple[int, int], list[tuple]] = {}
    for tree in model["trees"]:
        for kind, t, side in _components(tree):
            feats = _component_features(kind, t, side)
            if len(feats) == 0:
                base_points += _eval_component(kind, t, side, {})
            elif len(feats) == 1:
                single.setdefault(feats[0], []).append((kind, t, side))
            else:
                pairs.setdefault((feats[0], feats[1]), []).append((kind, t, side))

    main_effects = {}
    for f, comps in sorted(single.items()):
        thresholds = sorted(
            {t for kind, tree, side in comps for t in _component_thresholds(kind, tree, side, f)}
        )
        reps = _representatives(thresholds)
        bins = [
            {
                "interval": _interval_label(thresholds, i),
                "hi": thresholds[i] if i < len(thresholds) else None,  # bin = x <= hi
                "points_micro": sum(
                    _eval_component(kind, tree, side, {f: rep}) for kind, tree, side in comps
                ),
            }
            for i, rep in enumerate(reps)
        ]
        main_effects[names[f]] = {"feature_index": f, "thresholds": thresholds, "bins": bins}

    interactions = {}
    for (f, g), comps in sorted(pairs.items()):
        ts_f = sorted(
            {t for kind, tree, side in comps for t in _component_thresholds(kind, tree, side, f)}
        )
        ts_g = sorted(
            {t for kind, tree, side in comps for t in _component_thresholds(kind, tree, side, g)}
        )
        reps_f, reps_g = _representatives(ts_f), _representatives(ts_g)
        grid = [
            [
                sum(_eval_component(kind, tree, side, {f: rf, g: rg}) for kind, tree, side in comps)
                for rg in reps_g
            ]
            for rf in reps_f
        ]
        interactions[f"{names[f]} x {names[g]}"] = {
            "feature_indices": [f, g],
            "row_feature": names[f],
            "col_feature": names[g],
            "row_thresholds": ts_f,
            "col_thresholds": ts_g,
            "row_intervals": [_interval_label(ts_f, i) for i in range(len(reps_f))],
            "col_intervals": [_interval_label(ts_g, i) for i in range(len(reps_g))],
            "grid_micro": grid,
        }

    return {
        "base_points_micro": base_points,
        "main_effects": main_effects,
        "interactions": interactions,
        "micro_scale": int(model["micro_scale"]),
        "scale": int(artifact["scale"]),
        "input_precision": str(model.get("input_precision", "float64")),
        "feature_names": names,
        "n_trees": len(model["trees"]),
        "whitebox_max_depth": depth,
        "exact": True,
        "artifact_hash": artifact.get("artifact_hash"),
    }


def score_from_scorecard(scorecard: dict, row) -> int:
    """Re-derive ``raw_micro`` from the scorecard alone — the audit re-sum.

    Applies the artifact's input precision, then adds base points, one bin
    per feature with a main effect, and one grid cell per interaction.
    Equals the runtime's ``raw_micro`` exactly, by construction.
    """
    values = [float(v) for v in row]
    if scorecard["input_precision"] == "float32":
        values = [struct.unpack("<f", struct.pack("<f", v))[0] for v in values]

    total = int(scorecard["base_points_micro"])
    for effect in scorecard["main_effects"].values():
        x = values[effect["feature_index"]]
        total += int(effect["bins"][bisect_left(effect["thresholds"], x)]["points_micro"])
    for inter in scorecard["interactions"].values():
        f, g = inter["feature_indices"]
        r = bisect_left(inter["row_thresholds"], values[f])
        c = bisect_left(inter["col_thresholds"], values[g])
        total += int(inter["grid_micro"][r][c])
    return total
