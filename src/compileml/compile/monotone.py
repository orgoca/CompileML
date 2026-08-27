"""Independent verification of monotone feature constraints.

Trainer-agnostic by design: these checks read the *quantized integer
trees* — the thing that actually ships — never the fitted estimator. A
constraint holds because the compiled arithmetic says so, not because a
training library promised to enforce it.

Two layers:

- :func:`verify_monotone_constraints` — per-tree, any depth. A sum of
  monotone functions is monotone, so per-tree monotonicity is a
  *sufficient* condition for the whole model; and because
  ``HistGradientBoostingRegressor`` enforces constraints per tree during
  growth, models it trains never false-alarm here.
- :func:`scorecard_monotone_report` — aggregate, depth ≤ 2 only.
  Necessary *and* sufficient: it certifies the function a committee will
  actually read (the printed scorecard), via the worst-case increment
  ``Δmain + Σ min-over-partner-bins Δgrid ≥ 0`` per adjacent bin pair.

Everything here is exact integer arithmetic on ``value_micro`` leaves.
"""

from __future__ import annotations

from bisect import bisect_left
from itertools import product

LEAF = -2


def normalize_constraints(constraints, n_features: int, feature_names=None) -> list[int] | None:
    """Normalize to a list of -1/0/+1 per feature, or None if unconstrained.

    Accepts a positional sequence, a dict keyed by feature index, or —
    when ``feature_names`` is given — a dict keyed by feature name.
    """
    if constraints is None:
        return None
    if isinstance(constraints, dict):
        cst = [0] * n_features
        for key, sign in constraints.items():
            if isinstance(key, str):
                if feature_names is None:
                    raise ValueError(
                        f"constraint key {key!r} is a name, but no feature names are "
                        "available here — use feature indices, or pass the dict to "
                        "build_artifact where names are known"
                    )
                try:
                    idx = list(feature_names).index(key)
                except ValueError:
                    raise ValueError(f"unknown feature name in constraints: {key!r}") from None
            else:
                idx = int(key)
                if not 0 <= idx < n_features:
                    raise ValueError(
                        f"constraint index {idx} out of range for {n_features} features"
                    )
            cst[idx] = int(sign)
    else:
        cst = [int(v) for v in constraints]
        if len(cst) != n_features:
            raise ValueError(
                f"constraints length {len(cst)} does not match n_features {n_features}"
            )
    bad = sorted({v for v in cst if v not in (-1, 0, 1)})
    if bad:
        raise ValueError(f"constraint signs must be -1, 0 or +1; got {bad}")
    return cst if any(cst) else None


def _tree_thresholds_by_feature(tree: dict) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for f, t in zip(tree["feature"], tree["threshold"]):
        if f != LEAF:
            out.setdefault(int(f), []).append(float(t))
    return {f: sorted(set(ts)) for f, ts in out.items()}


def _representatives(thresholds: list[float]) -> list[float]:
    """One probe value per interval of ``(-inf, t1], …, (tk, inf)``."""
    if not thresholds:
        return [0.0]
    return [thresholds[0], *thresholds[1:], thresholds[-1] + 1.0]


def _eval_tree(tree: dict, values: dict[int, float]) -> int:
    node = 0
    while tree["feature"][node] != LEAF:
        f = tree["feature"][node]
        node = tree["left"][node] if values[f] <= tree["threshold"][node] else tree["right"][node]
    return int(tree["value_micro"][node])


def verify_monotone_constraints(model_int: dict, constraints) -> dict:
    """Check every quantized tree against the declared constraint signs.

    Exact and cheap at any depth: a tree references at most ``depth``
    features, so its cell grid is tiny. Per-tree monotonicity is
    sufficient for the whole model (monotone functions sum to a monotone
    function). Returns ``{"ok", "n_violations", "violations", "method"}``
    with at most 20 example violations, each pinpointing the tree, the
    feature, the fixed context, and the offending leaf-value sequence.
    """
    cst = normalize_constraints(constraints, int(model_int["n_features"]))
    if cst is None:
        return {"ok": True, "n_violations": 0, "violations": [], "method": "per_tree"}

    violations: list[dict] = []
    n_total = 0
    for t_idx, tree in enumerate(model_int["trees"]):
        thresholds = _tree_thresholds_by_feature(tree)
        constrained = [f for f in thresholds if cst[f] != 0]
        if not constrained:
            continue
        others = sorted(f for f in thresholds if cst[f] == 0)
        for f in constrained:
            sign = cst[f]
            f_reps = _representatives(thresholds[f])
            # Fix every OTHER feature of this tree (constrained ones included:
            # they get their own pass) and sweep f across its intervals.
            fixed = [g for g in constrained if g != f] + others
            fixed_reps = [_representatives(thresholds[g]) for g in fixed]
            for combo in product(*fixed_reps) if fixed else [()]:
                context = dict(zip(fixed, combo))
                seq = []
                for rep in f_reps:
                    context[f] = rep
                    seq.append(_eval_tree(tree, context))
                if any(sign * (b - a) < 0 for a, b in zip(seq, seq[1:])):
                    n_total += 1
                    if len(violations) < 20:
                        violations.append(
                            {
                                "tree": t_idx,
                                "feature": int(f),
                                "sign": int(sign),
                                "context": {int(k): float(v) for k, v in context.items() if k != f},
                                "leaf_values_micro": seq,
                            }
                        )
    return {
        "ok": n_total == 0,
        "n_violations": n_total,
        "violations": violations,
        "method": "per_tree",
    }


def scorecard_monotone_report(scorecard: dict, constraints) -> dict:
    """Aggregate (necessary-and-sufficient) check on a depth ≤ 2 scorecard.

    For each constrained feature *f* and each adjacent pair of its bins,
    the worst-case increment of the full function is the main-effect delta
    plus, for every interaction grid involving *f*, the minimum delta over
    the partner's bins. The constraint holds iff every worst-case
    increment carries the declared sign. This certifies the printed
    tables themselves — a validator can repeat it in a spreadsheet.
    """
    names = list(scorecard["feature_names"])
    cst = normalize_constraints(constraints, len(names), feature_names=names)
    if cst is None:
        return {"ok": True, "per_feature": {}, "method": "scorecard_aggregate"}

    per_feature: dict[str, dict] = {}
    for f_idx, sign in enumerate(cst):
        if sign == 0:
            continue
        name = names[f_idx]

        # This feature's global bin edges: union of its main-effect
        # thresholds and its thresholds inside every interaction it joins.
        edges: set[float] = set()
        main = scorecard["main_effects"].get(name)
        if main:
            edges.update(main["thresholds"])
        joined = []
        for inter in scorecard["interactions"].values():
            if f_idx in inter["feature_indices"]:
                axis = "row" if inter["feature_indices"][0] == f_idx else "col"
                joined.append((axis, inter))
                edges.update(inter[f"{axis}_thresholds"])
        if not edges:
            per_feature[name] = {"ok": True, "note": "feature unused by the model"}
            continue

        reps = _representatives(sorted(edges))

        def _main_at(rep: float, main=main) -> int:
            if not main:
                return 0
            return int(main["bins"][bisect_left(main["thresholds"], rep)]["points_micro"])

        worst = None
        ok = True
        for a, b in zip(reps, reps[1:]):
            increment = sign * (_main_at(b) - _main_at(a))
            for axis, inter in joined:
                grid = inter["grid_micro"]
                ia = bisect_left(inter[f"{axis}_thresholds"], a)
                ib = bisect_left(inter[f"{axis}_thresholds"], b)
                if axis == "row":
                    deltas = [sign * (grid[ib][c] - grid[ia][c]) for c in range(len(grid[0]))]
                else:
                    deltas = [sign * (row[ib] - row[ia]) for row in grid]
                increment += min(deltas)
            worst = increment if worst is None else min(worst, increment)
            if increment < 0:
                ok = False
        per_feature[name] = {"ok": ok, "worst_increment_micro": int(worst)}

    return {
        "ok": all(v["ok"] for v in per_feature.values()),
        "per_feature": per_feature,
        "method": "scorecard_aggregate",
    }
