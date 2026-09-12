"""Risk band construction from latent scores.

Edges are deduplicated at the artifact's fixed-point scale, so a builder
never hands ``build_artifact`` a ladder it will refuse.

Builders return a :class:`BandSpec` — float edges, labels, and metadata —
which ``compileml.artifact.build_artifact`` freezes into the fixed-point
integer ladder. Note: builders deliberately record **no timestamps**;
identical inputs must yield identical artifacts (and identical hashes).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression

from compileml.compile.quantize import rha

DEFAULT_SCALE = 1000  # matches build_artifact's default display scale


@dataclass
class BandSpec:
    """Float-space band definition produced by the builders."""

    edges: list[float]  # n_bands + 1, strictly increasing
    labels: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def n_bands(self) -> int:
        return len(self.edges) - 1


def _dedupe_at_scale(edges: np.ndarray, scale: int) -> np.ndarray:
    """Drop edges that collapse onto the same integer at the artifact scale.

    This used to nudge ties apart by 1e-12, which satisfies a float
    strictly-increasing check and then collides anyway inside
    ``build_artifact``, where edges become integers at ``scale``. A mass of
    identical latents produces exactly that, and it is the normal outcome on
    a low base rate: a squared-error regressor on a skewed target predicts
    below zero for a share of rows, and clipping pins them all to the same
    value. Above roughly ``1 / n_bands`` of the volume, at least one quantile
    cut lands inside the pinned mass.

    Dropping the offending cut yields fewer bands than requested, which the
    callers record and warn about. The alternative — an invalid ladder
    discovered two steps later — is worse.

    The top edge is *extended* rather than dropped, so the highest band keeps
    its rows and a score above anything seen still lands somewhere.
    """
    kept = [float(edges[0])]
    last = rha(float(edges[0]) * scale)
    for e in edges[1:-1]:
        value = rha(float(e) * scale)
        if value > last:
            kept.append(float(e))
            last = value
    top = float(edges[-1])
    if rha(top * scale) <= last:
        top = (last + 1) / scale
    kept.append(top)
    return np.asarray(kept, dtype=float)


def _quantile_edges(
    latent: np.ndarray, n_bands: int, method: str, scale: int = DEFAULT_SCALE
) -> np.ndarray:
    if method == "quantile":
        edges = np.quantile(latent, np.linspace(0.0, 1.0, n_bands + 1))
        edges[0] = float(np.min(latent))
        edges[-1] = float(np.max(latent)) + 1e-12
    elif method == "equal_width":
        edges = np.linspace(float(np.min(latent)), float(np.max(latent)), n_bands + 1)
        edges[-1] += 1e-12
    else:
        raise ValueError("method must be 'quantile' or 'equal_width'")
    return _dedupe_at_scale(edges, scale)


def _warn_reduced(requested: int, actual: int, scale: int) -> None:
    if actual >= requested:
        return
    warnings.warn(
        f"requested {requested} bands but the latent supports {actual} at "
        f"scale={scale}: {requested - actual} quantile cut(s) fell inside a mass of "
        "identical scores and were dropped, because edges that collide at fixed "
        "point cannot be compiled. A large pinned mass usually means the distilled "
        "latent is clipping — check share_outside_unit_interval — and the ladder "
        "degenerates once that mass exceeds about 1/n_bands of the volume.",
        stacklevel=3,
    )


def _labels(n: int) -> list[str]:
    return [f"G{i + 1:02d}" for i in range(n)]


def quantile_bands(
    latent, n_bands: int = 10, *, method: str = "quantile", scale: int = DEFAULT_SCALE
) -> BandSpec:
    """Plain quantile (or equal-width) bands; no outcome data required.

    ``scale`` is the artifact's display scale, and edges that would collide
    once converted to integers at that scale are dropped rather than emitted.
    The result can therefore carry fewer bands than requested; the reduction
    is warned about and recorded in ``metadata["requested_n_bands"]``.
    """
    x = np.asarray(latent, dtype=float).reshape(-1)
    edges = _quantile_edges(x, n_bands, method, scale)
    k = len(edges) - 1
    _warn_reduced(n_bands, k, scale)
    counts = np.bincount(np.clip(np.digitize(x, edges) - 1, 0, k - 1), minlength=k)
    return BandSpec(
        edges=[float(e) for e in edges],
        labels=_labels(k),
        metadata={
            "method": method,
            "counts": [int(c) for c in counts],
            "requested_n_bands": int(n_bands),
            "scale": int(scale),
        },
    )


def monotone_quantile_bands(
    latent,
    y,
    n_bands: int = 10,
    *,
    allow_merge: bool = False,
    merge_eps: float = 0.005,
    scale: int = DEFAULT_SCALE,
) -> BandSpec:
    """Quantile bands with empirical bad rates and isotonic-smoothed semantics.

    Fixed-K quantile edges by default. With ``allow_merge=True``, adjacent
    bands whose empirical bad rates invert by more than ``merge_eps`` are
    merged until no material violation remains — trading band count for
    guaranteed-monotone empirical semantics.

    ``scale`` is the artifact's display scale; edges that would collide once
    converted to integers there are dropped instead of emitted, so the result
    can carry fewer bands than requested. That reduction is separate from
    ``allow_merge``, which only merges bad-rate inversions.
    """
    F = np.asarray(latent, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if F.shape[0] != y_arr.shape[0]:
        raise ValueError("latent and y must have the same length")

    edges = _quantile_edges(F, n_bands, "quantile", scale)
    _warn_reduced(n_bands, len(edges) - 1, scale)

    def stats(cur_edges):
        idx = np.clip(np.digitize(F, cur_edges) - 1, 0, len(cur_edges) - 2)
        k = len(cur_edges) - 1
        counts = np.bincount(idx, minlength=k)
        bad = np.bincount(idx, weights=y_arr, minlength=k)
        rate = np.where(counts > 0, bad / np.maximum(counts, 1), float(np.mean(y_arr)))
        return counts.astype(int), rate

    merges: list[dict] = []
    counts, emp_rate = stats(edges)
    if allow_merge:
        while len(emp_rate) > 1:
            violations = emp_rate[:-1] - emp_rate[1:]
            worst = float(np.max(violations))
            if worst <= merge_eps:
                break
            i = int(np.argmax(violations))
            merges.append({"merge_idx": i, "violation": worst})
            edges = np.delete(edges, i + 1)
            counts, emp_rate = stats(edges)

    smoothed = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
        np.arange(len(emp_rate)), emp_rate
    )

    k = len(edges) - 1
    return BandSpec(
        edges=[float(e) for e in edges],
        labels=_labels(k),
        metadata={
            "method": "monotone_quantile",
            "counts": [int(c) for c in counts],
            "empirical_bad_rate": [float(v) for v in emp_rate],
            "smoothed_bad_rate": [float(v) for v in smoothed],
            "allow_merge": bool(allow_merge),
            "merge_eps": float(merge_eps),
            "merges": merges,
            "requested_n_bands": int(n_bands),
            "scale": int(scale),
        },
    )
