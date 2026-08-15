"""Search-and-certify band construction.

Two builders that *discover* the number of statistically defensible bands
instead of taking K on faith:

- :func:`semantic_bands` — merge/split search over Jeffreys PD confidence
  intervals with within-band AUC collapse tests, then a one-time bootstrap
  certification pass per final band.
- :func:`governance_bands` — Welch t-test separation of adjacent bands plus
  a within-band residual-AUC cap; the more conservative, committee-facing
  variant.

Both return a :class:`~compileml.bands.builders.BandSpec` whose metadata
carries the full statistical evidence (per-band PDs, CIs, AUCs, flags), so
the certification travels inside the hashed artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

from compileml.bands.builders import BandSpec


def _next_down(x: float) -> float:
    return float(np.nextafter(float(x), -np.inf))


def _next_up(x: float) -> float:
    return float(np.nextafter(float(x), np.inf))


def _labels(n: int) -> list[str]:
    return [f"G{i + 1:02d}" for i in range(n)]


def _auc_point(F_seg: np.ndarray, y_seg: np.ndarray) -> float:
    if len(y_seg) < 5 or len(np.unique(y_seg)) < 2:
        return 0.5
    return float(roc_auc_score(y_seg, F_seg))


def _make_proto_ranges(n: int, proto_min_size: int) -> list[tuple[int, int]]:
    proto_min_size = max(10, int(proto_min_size))
    ranges = []
    left = 0
    while left < n:
        right = min(n, left + proto_min_size)
        ranges.append((left, right))
        left = right
    return ranges


def _edges_from_slices(Fs: np.ndarray, slices) -> list[float]:
    edges = [_next_down(Fs[0])]
    for s in slices:
        edges.append(_next_up(Fs[s.R - 1]))
    for i in range(1, len(edges)):
        if not edges[i] > edges[i - 1]:
            edges[i] = _next_up(edges[i - 1])
    return edges


# ---------------------------------------------------------------------------
# semantic_bands — Jeffreys CI separation + AUC collapse + bootstrap certify
# ---------------------------------------------------------------------------


@dataclass
class _Stats:
    L: int
    R: int
    n: int
    y_sum: int
    pd_mean: float
    pd_ci: tuple[float, float]
    auc_hat: float


def _jeffreys_ci(y_sum: int, n: int, alpha: float) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    a, b = y_sum + 0.5, (n - y_sum) + 0.5
    lo = float(beta_dist.ppf(alpha / 2, a, b))
    hi = float(beta_dist.ppf(1 - alpha / 2, a, b))
    lo = 0.0 if np.isnan(lo) else min(max(lo, 0.0), 1.0)
    hi = 1.0 if np.isnan(hi) else min(max(hi, 0.0), 1.0)
    return (lo, hi)


def _compute(Fs, ys, L, R, alpha, cache) -> _Stats:
    key = (L, R)
    if key not in cache:
        y_seg, F_seg = ys[L:R], Fs[L:R]
        n = int(R - L)
        y_sum = int(y_seg.sum()) if n else 0
        cache[key] = _Stats(
            L,
            R,
            n,
            y_sum,
            float((y_sum + 0.5) / (n + 1.0)) if n else float("nan"),
            _jeffreys_ci(y_sum, n, alpha),
            _auc_point(F_seg, y_seg),
        )
    return cache[key]


def _boundary_ok(a: _Stats, b: _Stats, delta_sep: float) -> bool:
    if np.any(np.isnan(a.pd_ci)) or np.any(np.isnan(b.pd_ci)):
        return False
    return (b.pd_ci[0] - a.pd_ci[1]) >= float(delta_sep)


def _collapse_ok(s: _Stats, eps_auc: float) -> bool:
    return float(s.auc_hat) <= (0.5 + float(eps_auc) + 1e-12)


def _constraints(stats, min_band_size, delta_sep, eps_auc):
    diag = {"size": [], "collapse": [], "boundary": []}
    for i, s in enumerate(stats):
        if s.n < min_band_size:
            diag["size"].append(i)
        if not _collapse_ok(s, eps_auc):
            diag["collapse"].append(i)
    for i in range(len(stats) - 1):
        if not _boundary_ok(stats[i], stats[i + 1], delta_sep):
            diag["boundary"].append(i)
    ok = not (diag["size"] or diag["collapse"] or diag["boundary"])
    return ok, diag


def _merge_at(stats, i, Fs, ys, alpha, cache):
    merged = _compute(Fs, ys, stats[i].L, stats[i + 1].R, alpha, cache)
    return stats[:i] + [merged] + stats[i + 2 :]


def _choose_merge(stats, diag, Fs, ys, alpha, cache, eps_auc):
    K = len(stats)
    if diag["size"]:
        i = diag["size"][0]
        candidates = [c for c in (i - 1, i) if 0 <= c < K - 1]
        best, best_score = candidates[0], np.inf
        for m in candidates:
            merged = _compute(Fs, ys, stats[m].L, stats[m + 1].R, alpha, cache)
            score = (merged.pd_ci[1] - merged.pd_ci[0]) + 0.5 * max(
                0.0, merged.auc_hat - (0.5 + eps_auc)
            )
            if score < best_score:
                best, best_score = m, score
        return best
    if diag["boundary"]:
        return int(diag["boundary"][0])
    i = diag["collapse"][0]
    candidates = [c for c in (i - 1, i) if 0 <= c < K - 1]
    best, best_score = candidates[0], np.inf
    for m in candidates:
        merged = _compute(Fs, ys, stats[m].L, stats[m + 1].R, alpha, cache)
        if merged.auc_hat < best_score:
            best, best_score = m, merged.auc_hat
    return best


def _try_split(stats, i, Fs, ys, alpha, cache, min_band_size, delta_sep, eps_auc, n_candidates=5):
    s = stats[i]
    if s.n < 2 * min_band_size:
        return None
    left_nb = stats[i - 1] if i > 0 else None
    right_nb = stats[i + 1] if i < len(stats) - 1 else None
    cuts = np.unique(
        np.linspace(s.L + min_band_size, s.R - min_band_size, n_candidates).astype(int)
    )
    best_new, best_score = None, -np.inf
    for c in cuts:
        a = _compute(Fs, ys, s.L, int(c), alpha, cache)
        b = _compute(Fs, ys, int(c), s.R, alpha, cache)
        if a.n < min_band_size or b.n < min_band_size:
            continue
        if not (_collapse_ok(a, eps_auc) and _collapse_ok(b, eps_auc)):
            continue
        if not _boundary_ok(a, b, delta_sep):
            continue
        if left_nb is not None and not _boundary_ok(left_nb, a, delta_sep):
            continue
        if right_nb is not None and not _boundary_ok(b, right_nb, delta_sep):
            continue
        score = (b.pd_ci[0] - a.pd_ci[1]) - 0.1 * (max(a.auc_hat, b.auc_hat) - 0.5)
        if score > best_score:
            best_score, best_new = score, stats[:i] + [a, b] + stats[i + 1 :]
    return best_new


def _bootstrap_auc_ci(F_seg, y_seg, alpha, n_boot, rng):
    n = len(y_seg)
    if n < 5 or len(np.unique(y_seg)) < 2:
        return 0.5, (0.5, 0.5)
    auc_hat = float(roc_auc_score(y_seg, F_seg))
    boots = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        ys_b = y_seg[samp]
        boots[b] = 0.5 if len(np.unique(ys_b)) < 2 else float(roc_auc_score(ys_b, F_seg[samp]))
    lo = min(max(float(np.quantile(boots, alpha / 2)), 0.0), 1.0)
    hi = min(max(float(np.quantile(boots, 1 - alpha / 2)), 0.0), 1.0)
    return auc_hat, (lo, hi)


def semantic_bands(
    latent,
    y,
    max_bands: int = 10,
    min_band_size: int = 300,
    *,
    delta_sep: float = 0.002,
    eps_auc_search: float = 0.02,
    eps_auc_cert: float = 0.01,
    alpha: float = 0.05,
    proto_min_size: int = 200,
    split_passes: int = 2,
    n_boot_cert: int = 300,
    rng_seed: int = 7,
) -> BandSpec:
    """Discover the maximum number of statistically separable risk bands.

    A banding is accepted when every band has ``n >= min_band_size``, adjacent
    Jeffreys PD intervals are separated by at least ``delta_sep``, and no band
    retains internal rank-ordering power above ``0.5 + eps_auc`` (the score is
    "used up" within each band). Search runs on cheap point estimates; the
    final banding is certified once with bootstrap AUC intervals.
    """
    F = np.asarray(latent, dtype=float).reshape(-1)
    y_arr = np.asarray(y).reshape(-1).astype(int)
    if F.shape[0] != y_arr.shape[0]:
        raise ValueError("latent and y must have the same length")
    n = len(F)
    if n < 2 * min_band_size:
        raise ValueError("not enough samples for two bands at min_band_size")

    order = np.argsort(F)
    Fs, ys = F[order], y_arr[order]
    cache: dict = {}
    params = {
        "max_bands": int(max_bands),
        "min_band_size": int(min_band_size),
        "delta_sep": float(delta_sep),
        "eps_auc_search": float(eps_auc_search),
        "eps_auc_cert": float(eps_auc_cert),
        "alpha": float(alpha),
        "proto_min_size": int(proto_min_size),
        "split_passes": int(split_passes),
        "n_boot_cert": int(n_boot_cert),
        "rng_seed": int(rng_seed),
    }

    # Protos never start below min_band_size: sub-size protos make every band
    # violate the size constraint at once, and the greedy "merge the first
    # violator into the tighter-CI neighbor" rule then snowballs the leftmost
    # band through the whole distribution. Clamping removes the pathology and
    # is also faster (fewer protos).
    proto = max(int(proto_min_size), int(min_band_size))
    stats = [_compute(Fs, ys, L, R, alpha, cache) for L, R in _make_proto_ranges(n, proto)]

    # Merge until constraints hold (search phase, point estimates only).
    for _ in range(50_000):
        ok, diag = _constraints(stats, min_band_size, delta_sep, eps_auc_search)
        if ok or len(stats) <= 1:
            break
        stats = _merge_at(
            stats,
            _choose_merge(stats, diag, Fs, ys, alpha, cache, eps_auc_search),
            Fs,
            ys,
            alpha,
            cache,
        )
    else:
        raise RuntimeError("merge search did not converge")

    if len(stats) < 2:
        s = _compute(Fs, ys, 0, n, alpha, cache)
        return BandSpec(
            edges=[_next_down(Fs[0]), _next_up(Fs[-1])],
            labels=["G01"],
            metadata={
                "method": "semantic_risk_bands",
                "expressible_n_bands": 1,
                "band_pd": [s.pd_mean],
                "band_pd_ci": [list(s.pd_ci)],
                "flags": {"no_discrete_classes": True},
                "params": params,
            },
        )

    # Split refinement.
    for _ in range(max(0, int(split_passes))):
        improved, i = False, 0
        while i < len(stats):
            attempt = _try_split(
                stats, i, Fs, ys, alpha, cache, min_band_size, delta_sep, eps_auc_search
            )
            if attempt is not None:
                stats, improved, i = attempt, True, max(0, i - 1)
            else:
                i += 1
        if not improved:
            break

    k_star = len(stats)
    cap_binding = k_star > int(max_bands)
    while len(stats) > int(max_bands):
        best_m, best_damage = 0, np.inf
        for m in range(len(stats) - 1):
            merged = _compute(Fs, ys, stats[m].L, stats[m + 1].R, alpha, cache)
            damage = (merged.pd_ci[1] - merged.pd_ci[0]) + max(
                0.0, merged.auc_hat - (0.5 + eps_auc_search)
            )
            if damage < best_damage:
                best_m, best_damage = m, damage
        stats = _merge_at(stats, best_m, Fs, ys, alpha, cache)

    # One-time bootstrap certification.
    rng = np.random.default_rng(rng_seed)
    band_auc, band_auc_ci = [], []
    for s in stats:
        auc_hat, ci = _bootstrap_auc_ci(Fs[s.L : s.R], ys[s.L : s.R], alpha, n_boot_cert, rng)
        band_auc.append(auc_hat)
        band_auc_ci.append([ci[0], ci[1]])

    sep_margins = [float(stats[i + 1].pd_ci[0] - stats[i].pd_ci[1]) for i in range(len(stats) - 1)]
    return BandSpec(
        edges=_edges_from_slices(Fs, stats),
        labels=_labels(len(stats)),
        metadata={
            "method": "semantic_risk_bands",
            "expressible_n_bands": int(k_star),
            "band_pd": [s.pd_mean for s in stats],
            "band_pd_ci": [list(s.pd_ci) for s in stats],
            "band_auc": band_auc,
            "band_auc_ci": band_auc_ci,
            "sep_margins": sep_margins,
            "flags": {
                "cap_binding": bool(cap_binding),
                "underexpressive": bool(k_star < int(max_bands)),
                "boundary_overlap_present": any(m < delta_sep for m in sep_margins),
                "residual_rank_present": any(
                    ci[1] > (0.5 + eps_auc_cert + 1e-12) for ci in band_auc_ci
                ),
            },
            "params": params,
        },
    )


# ---------------------------------------------------------------------------
# governance_bands — Welch t-test separation + within-band AUC cap
# ---------------------------------------------------------------------------


@dataclass
class _GovStats:
    L: int
    R: int
    n: int
    pd: float
    auc: float


def _gov_compute(Fs, ys, L, R, cache) -> _GovStats:
    key = (L, R)
    if key not in cache:
        y_seg, F_seg = ys[L:R], Fs[L:R]
        n = int(R - L)
        cache[key] = _GovStats(
            L, R, n, float(np.mean(y_seg)) if n else float("nan"), _auc_point(F_seg, y_seg)
        )
    return cache[key]


def _gov_ttest_p(ys, a: _GovStats, b: _GovStats) -> float:
    ya, yb = ys[a.L : a.R], ys[b.L : b.R]
    if len(ya) < 2 or len(yb) < 2:
        return 1.0
    p = float(ttest_ind(ya, yb, equal_var=False).pvalue)
    return p if np.isfinite(p) else 1.0


def governance_bands(
    latent,
    y,
    max_bands: int = 10,
    min_band_size: int = 2000,
    *,
    proto_min_size: int = 500,
    auc_max_within: float = 0.60,
    alpha_sep: float = 0.05,
    min_delta_pd: float = 0.0,
    enforce_monotone_pd: bool = True,
) -> BandSpec:
    """Committee-facing bands: adjacent-band Welch t-test separation, a
    within-band residual-AUC cap, and (optionally) strictly monotone PDs.

    Discovers the maximum defensible band count and applies ``max_bands``
    as a ceiling only.
    """
    F = np.asarray(latent, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if F.shape[0] != y_arr.shape[0]:
        raise ValueError("latent and y must have the same length")
    n = len(F)
    if n < 2 * min_band_size:
        raise ValueError("not enough samples for two bands at min_band_size")

    order = np.argsort(F)
    Fs, ys = F[order], y_arr[order]
    cache: dict = {}

    # Same proto clamp as semantic_bands: sub-size protos trigger a greedy
    # left-band snowball through the size-violation branch.
    proto = max(int(proto_min_size), int(min_band_size))
    stats = [_gov_compute(Fs, ys, L, R, cache) for L, R in _make_proto_ranges(n, proto)]

    def diagnose(sts):
        diag = {"size": [], "auc": [], "sep": [], "mono": []}
        for i, s in enumerate(sts):
            if s.n < min_band_size:
                diag["size"].append(i)
            if s.auc > auc_max_within + 1e-12:
                diag["auc"].append(i)
        for i in range(len(sts) - 1):
            a, b = sts[i], sts[i + 1]
            if enforce_monotone_pd and not (b.pd >= a.pd - 1e-15):
                diag["mono"].append(i)
            if (_gov_ttest_p(ys, a, b) > alpha_sep) or ((b.pd - a.pd) < min_delta_pd):
                diag["sep"].append(i)
        return diag

    for _ in range(200_000):
        diag = diagnose(stats)
        if not any(diag.values()) or len(stats) <= 1:
            break
        if diag["size"]:
            m = max(0, min(diag["size"][0], len(stats) - 2))
        elif diag["mono"]:
            m = diag["mono"][0]
        elif diag["sep"]:
            m = diag["sep"][0]
        else:
            m = max(0, min(diag["auc"][0], len(stats) - 2))
        merged = _gov_compute(Fs, ys, stats[m].L, stats[m + 1].R, cache)
        stats = stats[:m] + [merged] + stats[m + 2 :]
    else:
        raise RuntimeError("merge search did not converge")

    k_star = len(stats)
    cap_binding = k_star > int(max_bands)
    while len(stats) > int(max_bands):
        gaps = [abs(stats[i + 1].pd - stats[i].pd) for i in range(len(stats) - 1)]
        m = int(np.argmin(gaps))
        merged = _gov_compute(Fs, ys, stats[m].L, stats[m + 1].R, cache)
        stats = stats[:m] + [merged] + stats[m + 2 :]

    adj_p = [_gov_ttest_p(ys, stats[i], stats[i + 1]) for i in range(len(stats) - 1)]
    adj_delta = [float(stats[i + 1].pd - stats[i].pd) for i in range(len(stats) - 1)]

    return BandSpec(
        edges=_edges_from_slices(Fs, stats),
        labels=_labels(len(stats)),
        metadata={
            "method": "governance_bands",
            "expressible_n_bands": int(k_star),
            "band_pd": [s.pd for s in stats],
            "band_auc": [s.auc for s in stats],
            "adjacent_p_ttest": adj_p,
            "adjacent_delta_pd": adj_delta,
            "flags": {
                "cap_binding": bool(cap_binding),
                "underexpressive": bool(k_star < int(max_bands)),
                "no_discrete_classes": bool(k_star < 2),
            },
            "params": {
                "max_bands": int(max_bands),
                "min_band_size": int(min_band_size),
                "proto_min_size": int(proto_min_size),
                "auc_max_within": float(auc_max_within),
                "alpha_sep": float(alpha_sep),
                "min_delta_pd": float(min_delta_pd),
                "enforce_monotone_pd": bool(enforce_monotone_pd),
            },
        },
    )
