"""Per-section fairness metrics, computed from decisions rather than models.

Every function here takes ``decide()`` payloads — the integers a production
decision actually carried — and never recomputes the model. That is the same
rule ``compileml.viz`` follows, and it is why this describes the deployed
decision rather than a reconstruction of it.

Nothing here depends on SHAP. Two sections are better for it: the attribution
decomposition (§6) sums to the group gap exactly rather than approximately,
and reason parity (§10) audits the codes an applicant would actually be sent
rather than a ranking that stands in for them.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

# Approval convention: a higher latent means higher risk, so an applicant is
# approved when their score falls *below* the cutoff.
RISK_INCREASES_WITH_SCORE = True


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sanely on small groups, unlike normal.

    A fairness report is read group by group, and the smallest group is the
    one most likely to drive a finding. A normal-approximation interval on
    forty applicants produces confident nonsense; this does not.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def _groups(protected, labels=None):
    values = sorted({v for v in np.asarray(protected).tolist()})
    labels = labels or {v: str(v) for v in values}
    return [(v, str(labels.get(v, v))) for v in values]


# ---------------------------------------------------------------- layer 1
def representation(y, protected, *, labels=None, X=None, feature_names=None) -> dict:
    """§1 — who is in the population, and what their observed rates are.

    Also reports per-feature distribution divergence when ``X`` is given, so a
    reviewer can see which *inputs* differ before asking why scores do. A
    score gap explained entirely by an input gap is a different conversation
    from one that is not.
    """
    from scipy.stats import ks_2samp

    y = np.asarray(y, dtype=int).reshape(-1)
    g = np.asarray(protected).reshape(-1)
    out: dict = {"groups": {}, "n_total": int(y.size)}
    for value, name in _groups(g, labels):
        mask = g == value
        n = int(mask.sum())
        bad = int(y[mask].sum())
        lo, hi = wilson_interval(bad, n)
        out["groups"][name] = {
            "n": n,
            "share": float(n / y.size) if y.size else float("nan"),
            "base_rate": float(bad / n) if n else float("nan"),
            "base_rate_ci": [lo, hi],
        }

    if X is not None and feature_names is not None:
        X = np.asarray(X, dtype=float)
        pairs = _groups(g, labels)
        if len(pairs) == 2:
            (va, _), (vb, _) = pairs
            rows = []
            for j, fname in enumerate(feature_names):
                a, b = X[g == va, j], X[g == vb, j]
                if a.size and b.size:
                    stat = float(ks_2samp(a, b).statistic)
                    rows.append(
                        {
                            "feature": str(fname),
                            "ks": stat,
                            "mean_a": float(a.mean()),
                            "mean_b": float(b.mean()),
                        }
                    )
            out["feature_divergence"] = sorted(rows, key=lambda r: -r["ks"])
    return out


def score_distribution(decisions, protected, *, labels=None) -> dict:
    """§2 — group score distributions, KS and Wasserstein."""
    from scipy.stats import ks_2samp, wasserstein_distance

    g = np.asarray(protected).reshape(-1)
    latent = np.array([d["latent_int"] for d in decisions], dtype=float)
    out: dict = {"groups": {}}
    for value, name in _groups(g, labels):
        s = latent[g == value]
        out["groups"][name] = {
            "mean": float(s.mean()) if s.size else float("nan"),
            "std": float(s.std()) if s.size else float("nan"),
            "median": float(np.median(s)) if s.size else float("nan"),
        }
    pairs = _groups(g, labels)
    if len(pairs) == 2:
        a, b = latent[g == pairs[0][0]], latent[g == pairs[1][0]]
        res = ks_2samp(a, b)
        out["ks"] = float(res.statistic)
        out["ks_pvalue"] = float(res.pvalue)
        out["wasserstein"] = float(wasserstein_distance(a, b))
    return out


def approval_rates(decisions, protected, threshold_int: int, *, labels=None) -> dict:
    """§3 — approval rate per group, with the adverse impact ratio.

    ``threshold_int`` is a cutoff in display-scale latent units; an applicant
    is approved below it. The ratio is reported against the four-fifths rule's
    0.80–1.25 window, and the window is reported alongside rather than
    collapsed into a pass/fail — a ratio of 0.81 is not the same finding as
    1.00 and should not print as if it were.
    """
    g = np.asarray(protected).reshape(-1)
    latent = np.array([d["latent_int"] for d in decisions], dtype=float)
    approved = latent < threshold_int

    out: dict = {"threshold_int": int(threshold_int), "groups": {}}
    rates = []
    for value, name in _groups(g, labels):
        mask = g == value
        n = int(mask.sum())
        k = int(approved[mask].sum())
        lo, hi = wilson_interval(k, n)
        rate = float(k / n) if n else float("nan")
        out["groups"][name] = {"n": n, "approved": k, "rate": rate, "rate_ci": [lo, hi]}
        rates.append((name, rate))

    if len(rates) == 2 and rates[1][1] > 0:
        ratio = rates[0][1] / rates[1][1]
        out["adverse_impact_ratio"] = float(ratio)
        out["ratio_of"] = f"{rates[0][0]} / {rates[1][0]}"
        out["within_four_fifths"] = bool(0.80 <= ratio <= 1.25)
    return out


# ---------------------------------------------------------------- layer 2
def calibration_by_group(decisions, y, protected, *, labels=None, n_bins: int = 10) -> dict:
    """§4 — is the calibrated PD equally honest per group?

    A model can pass every outcome test and still be systematically
    over-predicting risk for one group. That is its own finding, and it is
    invisible at the outcome layer.
    """
    g = np.asarray(protected).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    pd_hat = np.array([d["pd_ppm"] for d in decisions], dtype=float) / 1e6

    out: dict = {"groups": {}}
    for value, name in _groups(g, labels):
        mask = g == value
        p, obs = pd_hat[mask], y[mask]
        if p.size == 0:
            continue
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
        bins = []
        for b in range(max(len(edges) - 1, 1)):
            sel = idx == b
            if not sel.any():
                continue
            bins.append(
                {
                    "predicted": float(p[sel].mean()),
                    "observed": float(obs[sel].mean()),
                    "n": int(sel.sum()),
                }
            )
        gap = float(np.mean([abs(b["predicted"] - b["observed"]) for b in bins])) if bins else 0.0
        out["groups"][name] = {
            "mean_predicted": float(p.mean()),
            "mean_observed": float(obs.mean()),
            "bias": float(p.mean() - obs.mean()),
            "mean_abs_calibration_gap": gap,
            "bins": bins,
        }
    return out


def error_rates(decisions, y, protected, threshold_int: int, *, labels=None) -> dict:
    """§5 — FPR, FNR, precision and recall per group."""
    g = np.asarray(protected).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    latent = np.array([d["latent_int"] for d in decisions], dtype=float)
    flagged = latent >= threshold_int  # predicted "bad"

    out: dict = {"groups": {}}
    for value, name in _groups(g, labels):
        m = g == value
        yy, pp = y[m], flagged[m]
        tp = int((pp & (yy == 1)).sum())
        fp = int((pp & (yy == 0)).sum())
        fn = int((~pp & (yy == 1)).sum())
        tn = int((~pp & (yy == 0)).sum())
        out["groups"][name] = {
            "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
            "fnr": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
            "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
            "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "n": int(m.sum()),
        }
    return out


# ---------------------------------------------------------------- layer 3
def attribution_disparity(decisions, protected, feature_names, *, labels=None) -> dict:
    """§6 — decompose the mean group score gap by feature, exactly.

    The headline capability, and the reason this module is worth having in
    CompileML rather than taken from a general fairness library. Because the
    artifact's attribution reconciles to the score with a zero residual at
    depth ≤ 2, the per-feature contributions to a group gap **sum to the
    gap** — not approximately, exactly, in integer units.

    A share above 100% is a real finding rather than an error: one driver
    widens the gap further than observed while others partially offset it.
    That statement is unavailable from a decomposition whose parts do not sum
    to the whole.

    Requires ``decide(..., include_contributions=True)``.
    """
    if "contributions" not in decisions[0]:
        raise ValueError(
            "attribution_disparity needs per-feature contributions; call "
            "decide(artifact, row, include_contributions=True). The default "
            "payload carries 'attribution' as a method label only."
        )
    residual = max(abs(int(d["attribution_residual_half_micro"])) for d in decisions)
    if residual != 0:
        raise ValueError(
            f"attribution residual is {residual}, not zero: this artifact's "
            "attribution is not exact (whitebox depth > 2), so a decomposition "
            "would not sum to the gap. Compile at depth <= 2, or read the "
            "outcome-level sections only."
        )

    g = np.asarray(protected).reshape(-1)
    p = len(feature_names)
    contrib = np.zeros((len(decisions), p), dtype=np.int64)
    for i, d in enumerate(decisions):
        for c in d["contributions"]:
            contrib[i, c["index"]] = int(c["impact_half_micro"])
    movement = np.array(
        [2 * (int(d["raw_micro"]) - int(d["baseline_micro"])) for d in decisions],
        dtype=np.int64,
    )

    pairs = _groups(g, labels)
    if len(pairs) != 2:
        raise ValueError("attribution_disparity compares exactly two groups")
    (va, na), (vb, nb) = pairs
    ma, mb = g == va, g == vb

    gap = float(movement[ma].mean() - movement[mb].mean())
    per = contrib[ma].mean(0) - contrib[mb].mean(0)
    total = float(per.sum())

    rows = [
        {
            "feature": str(feature_names[j]),
            "gap_half_micro": float(per[j]),
            "share_pct": float(100 * per[j] / total) if total else float("nan"),
        }
        for j in range(p)
    ]
    return {
        "comparison": f"{na} minus {nb}",
        "mean_gap_half_micro": gap,
        "sum_of_feature_gaps": total,
        "residual": float(gap - total),
        "by_feature": sorted(rows, key=lambda r: -abs(r["gap_half_micro"])),
    }


def feature_swing(
    artifact, X, protected, *, labels=None, max_rows: int = 400, seed: int = 7
) -> dict:
    """§7 — how far each feature *can* move a score, per group.

    An infinitesimal derivative is the wrong instrument here. A compiled
    artifact is piecewise constant, so a small perturbation returns zero
    almost everywhere and a gradient-style sensitivity measures nothing but
    whether a threshold happened to fall nearby.

    What is meaningful for a step function is the **swing**: hold the rest of
    the row fixed, move one feature across the values the population actually
    takes, and record how far the score travels. Evaluated at the artifact's
    own split thresholds, so it is exact rather than sampled — the score
    between two thresholds is constant by construction.
    """
    from compileml.runtime.score import score_micro

    model = artifact["model"]
    names = artifact["features"]["names"]
    X = np.asarray(X, dtype=float)
    g = np.asarray(protected).reshape(-1)

    thresholds: dict[int, list[float]] = {}
    for tree in model["trees"]:
        for f, t in zip(tree["feature"], tree["threshold"]):
            if int(f) >= 0:
                thresholds.setdefault(int(f), []).append(float(t))
    probes = {
        j: sorted({*ts, max(ts) + 1.0}) if (ts := sorted(set(v))) else []
        for j, v in thresholds.items()
    }

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(X), size=min(max_rows, len(X)), replace=False)

    swing = np.zeros((len(pick), len(names)), dtype=float)
    for r, i in enumerate(pick):
        row = [float(v) for v in X[i]]
        for j, ps in probes.items():
            if not ps:
                continue
            original = row[j]
            scores = []
            for probe in ps:
                row[j] = probe
                scores.append(score_micro(model, row))
            row[j] = original
            swing[r, j] = float(max(scores) - min(scores))

    gp = g[pick]
    out: dict = {"n_rows": int(len(pick)), "groups": {}, "by_feature": []}
    per_group = {}
    for value, name in _groups(g, labels):
        m = gp == value
        per_group[name] = swing[m].mean(0) if m.any() else np.full(len(names), np.nan)
        out["groups"][name] = {
            "mean_swing_micro": float(swing[m].mean()) if m.any() else float("nan")
        }
    gnames = list(per_group)
    for j, fname in enumerate(names):
        row = {"feature": str(fname)}
        for name in gnames:
            row[name] = float(per_group[name][j])
        if len(gnames) == 2:
            a, b = per_group[gnames[0]][j], per_group[gnames[1]][j]
            row["ratio"] = float(a / b) if b else float("nan")
        out["by_feature"].append(row)
    out["by_feature"].sort(key=lambda r: -max(r.get(n, 0) or 0 for n in gnames))
    return out


def attribution_concentration(decisions, protected, feature_names, *, labels=None) -> dict:
    """§8 — how many drivers it takes to explain a decision, per group.

    This section has been reformulated twice, and both changes were forced by
    the object rather than by taste.

    It began as **curvature**, which is the right instrument for a smooth
    compiling function with real manifold structure, where a Hessian means
    something. A depth-2 ensemble is piecewise constant: second derivatives
    vanish almost everywhere and finite differences measure noise.

    It was then **interaction share** — main-effect points versus pairwise
    grid points, read off the scorecard. That is exact, and on real artifacts
    it is useless: a depth-2 ensemble trained on real data produced *zero*
    main effects and forty-seven interaction grids, because no tree happened
    to split on a single feature. The share is then 100% for everyone, which
    describes the model and says nothing about any group.

    What does vary by group, and answers the original question, is how
    **concentrated** a decision's explanation is. A row whose score movement
    comes overwhelmingly from one driver is legible and defensible; one spread
    thinly across six is neither, and it is harder to write an adverse-action
    notice for. Reported as the top driver's share and as an effective number
    of drivers, ``exp(entropy)``, which reads directly: "this group's
    decisions are explained by 3.2 features on average, that group's by 4.1."

    Uses the same exact contributions as §6, so it needs no scorecard and
    inherits the zero-residual guarantee.
    """
    if "contributions" not in decisions[0]:
        raise ValueError(
            "attribution_concentration needs per-feature contributions; call "
            "decide(artifact, row, include_contributions=True)."
        )
    g = np.asarray(protected).reshape(-1)
    p = len(feature_names)
    mag = np.zeros((len(decisions), p), dtype=float)
    for i, d in enumerate(decisions):
        for c in d["contributions"]:
            mag[i, c["index"]] = abs(int(c["impact_half_micro"]))

    total = mag.sum(axis=1)
    live = total > 0
    share = np.zeros_like(mag)
    share[live] = mag[live] / total[live, None]

    top1 = np.where(live, share.max(axis=1), np.nan)
    # log(1) = 0, so substituting 1 for the empty cells contributes nothing to
    # the sum and avoids evaluating log on zeros at all.
    logs = np.log(np.where(share > 0, share, 1.0))
    entropy = -(share * logs).sum(axis=1)
    effective = np.where(live, np.exp(entropy), np.nan)
    herfindahl = np.where(live, (share**2).sum(axis=1), np.nan)

    out: dict = {"n_features": p, "n_rows_without_movement": int((~live).sum()), "groups": {}}
    for value, name in _groups(g, labels):
        m = g == value
        # A group in which no decision moved off the baseline has nothing to
        # concentrate; say so rather than averaging an empty slice.
        usable = m & live
        entry = {"n": int(m.sum()), "n_with_movement": int(usable.sum())}
        if usable.any():
            entry |= {
                "top_driver_share": float(top1[usable].mean()),
                "effective_drivers": float(effective[usable].mean()),
                "herfindahl": float(herfindahl[usable].mean()),
            }
        else:
            entry |= {
                "top_driver_share": float("nan"),
                "effective_drivers": float("nan"),
                "herfindahl": float("nan"),
            }
        out["groups"][name] = entry
    names_seen = list(out["groups"])
    if len(names_seen) == 2:
        a, b = (out["groups"][n]["effective_drivers"] for n in names_seen)
        out["effective_drivers_ratio"] = float(a / b) if b else float("nan")
        out["comparison"] = f"{names_seen[0]} / {names_seen[1]}"
    return out


def model_interaction_structure(artifact) -> dict:
    """Model-level context for §8: how much of the card is pairwise at all.

    Not a group metric — it cannot be, since it is a property of the compiled
    model. Reported alongside §8 because "every component is an interaction"
    is the fact that makes a per-group interaction share meaningless, and a
    reader deserves to see it rather than wonder.
    """
    from compileml.scorecard import build_scorecard

    card = build_scorecard(artifact)
    n_main, n_inter = len(card["main_effects"]), len(card["interactions"])
    return {
        "n_main_effects": n_main,
        "n_interactions": n_inter,
        "all_pairwise": n_main == 0,
        "note": (
            (
                "no single-feature components: every part of this model is a pairwise "
                "interaction, so explanation complexity varies by row rather than by "
                "which kind of component fired"
            )
            if n_main == 0
            else ""
        ),
    }


def boundary_fragility(
    decisions, protected, threshold_int: int, *, labels=None, near: int = 25
) -> dict:
    """§9 — how close each group sits to the cutoff.

    An adverse impact ratio is a snapshot. This is its derivative: a group
    clustered near the boundary is one policy tweak away from a disparate
    outcome even when today's ratio passes comfortably, and that exposure is
    invisible to every outcome-level test.
    """
    g = np.asarray(protected).reshape(-1)
    latent = np.array([d["latent_int"] for d in decisions], dtype=float)
    distance = np.abs(latent - threshold_int)

    out: dict = {"threshold_int": int(threshold_int), "near_band": int(near), "groups": {}}
    for value, name in _groups(g, labels):
        d = distance[g == value]
        if d.size == 0:
            continue
        out["groups"][name] = {
            "mean_distance": float(d.mean()),
            "median_distance": float(np.median(d)),
            "share_within_band": float((d <= near).mean()),
            "n": int(d.size),
        }
    return out


def reason_parity(decisions, protected, *, labels=None, top_n: int = 5) -> dict:
    """§10 — are the same reasons cited to both groups?

    Regulatory rather than statistical: this is about what applicants are
    *told*. A systematic difference in cited reasons is a finding even when
    approval rates match exactly.

    Note what is being counted. ``reasons_negative`` is the adverse-action
    code the applicant would actually be sent, produced by the same runtime
    that produced the decision — not a feature-importance ranking standing in
    for one. The audit is of disclosed reasons, not of a proxy for them.
    """
    g = np.asarray(protected).reshape(-1)
    out: dict = {"groups": {}, "n_without_adverse_reason": 0}
    tallies = {}
    for value, name in _groups(g, labels):
        counter: Counter = Counter()
        n_rows = 0
        for i in np.where(g == value)[0]:
            reasons = decisions[i].get("reasons_negative") or []
            if not reasons:
                out["n_without_adverse_reason"] += 1
                continue
            counter[reasons[0]["feature"]] += 1
            n_rows += 1
        tallies[name] = (counter, n_rows)
        out["groups"][name] = {
            "n_with_reason": n_rows,
            "top": [
                {"feature": f, "share": float(c / n_rows) if n_rows else float("nan")}
                for f, c in counter.most_common(top_n)
            ],
        }

    names = list(tallies)
    if len(names) == 2:
        (ca, na), (cb, nb) = tallies[names[0]], tallies[names[1]]
        keys = set(ca) | set(cb)
        deltas = []
        for k in keys:
            sa = ca[k] / na if na else 0.0
            sb = cb[k] / nb if nb else 0.0
            deltas.append({"feature": k, names[0]: sa, names[1]: sb, "delta": float(sa - sb)})
        deltas.sort(key=lambda r: -abs(r["delta"]))
        out["divergence"] = deltas
        # Total variation distance between the two reason distributions.
        out["total_variation"] = float(0.5 * sum(abs(d["delta"]) for d in deltas))
    return out


def counterfactual(artifact, X, protected, protected_feature: str | None, *, labels=None) -> dict:
    """§11 — flip the protected attribute and see whether decisions move.

    When the attribute is not a model input this does not apply, and that is
    reported rather than skipped: "the model does not use it, so the test is
    inapplicable" is the desired outcome and should appear in the report as a
    positive statement rather than a silent gap.
    """
    names = list(artifact["features"]["names"])
    if protected_feature is None or protected_feature not in names:
        return {
            "applicable": False,
            "reason": (
                f"{protected_feature!r} is not among the artifact's features, so the "
                "model cannot be using it directly. The counterfactual test does not "
                "apply — which is the desired outcome, not a gap in the audit."
            ),
        }

    from compileml.runtime.bands import band_index
    from compileml.runtime.score import score_micro

    j = names.index(protected_feature)
    model = artifact["model"]
    edges = artifact["bands"]["edges_int"]
    scale, micro = int(artifact["scale"]), int(model["micro_scale"])
    ratio = micro // scale

    X = np.asarray(X, dtype=float)
    g = np.asarray(protected).reshape(-1)
    values = sorted({float(v) for v in np.unique(X[:, j])})
    if len(values) != 2:
        return {
            "applicable": False,
            "reason": (
                f"{protected_feature!r} takes {len(values)} values; " "the flip is defined for two."
            ),
        }
    other = {values[0]: values[1], values[1]: values[0]}

    changed, deltas = 0, []
    for row in X:
        r = [float(v) for v in row]
        before = score_micro(model, r)
        r[j] = other[float(row[j])]
        after = score_micro(model, r)
        deltas.append(after - before)
        b0 = band_index(max(0, min(before, micro)) // ratio, edges)
        b1 = band_index(max(0, min(after, micro)) // ratio, edges)
        changed += int(b0 != b1)

    deltas_arr = np.asarray(deltas, dtype=float)
    out: dict = {
        "applicable": True,
        "feature": protected_feature,
        "band_flip_rate": float(changed / len(X)),
        "mean_abs_score_change_micro": float(np.abs(deltas_arr).mean()),
        "groups": {},
    }
    for value, name in _groups(g, labels):
        m = g == value
        out["groups"][name] = {
            "mean_score_change_micro": float(deltas_arr[m].mean()) if m.any() else float("nan")
        }
    return out
