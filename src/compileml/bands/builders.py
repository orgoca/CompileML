"""Risk band construction from latent scores.

Builders return a :class:`BandSpec` — float edges, labels, and metadata —
which ``compileml.artifact.build_artifact`` freezes into the fixed-point
integer ladder. Note: builders deliberately record **no timestamps**;
identical inputs must yield identical artifacts (and identical hashes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class BandSpec:
    """Float-space band definition produced by the builders."""

    edges: list[float]  # n_bands + 1, strictly increasing
    labels: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def n_bands(self) -> int:
        return len(self.edges) - 1


def _quantile_edges(latent: np.ndarray, n_bands: int, method: str) -> np.ndarray:
    if method == "quantile":
        edges = np.quantile(latent, np.linspace(0.0, 1.0, n_bands + 1))
        edges[0] = float(np.min(latent))
        edges[-1] = float(np.max(latent)) + 1e-12
    elif method == "equal_width":
        edges = np.linspace(float(np.min(latent)), float(np.max(latent)), n_bands + 1)
        edges[-1] += 1e-12
    else:
        raise ValueError("method must be 'quantile' or 'equal_width'")
    # enforce strict monotonicity on degenerate distributions
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges


def _labels(n: int) -> list[str]:
    return [f"G{i + 1:02d}" for i in range(n)]


def quantile_bands(latent, n_bands: int = 10, *, method: str = "quantile") -> BandSpec:
    """Plain quantile (or equal-width) bands; no outcome data required."""
    x = np.asarray(latent, dtype=float).reshape(-1)
    edges = _quantile_edges(x, n_bands, method)
    counts = np.bincount(np.clip(np.digitize(x, edges) - 1, 0, n_bands - 1), minlength=n_bands)
    return BandSpec(
        edges=[float(e) for e in edges],
        labels=_labels(n_bands),
        metadata={"method": method, "counts": [int(c) for c in counts]},
    )


def monotone_quantile_bands(
    latent,
    y,
    n_bands: int = 10,
    *,
    allow_merge: bool = False,
    merge_eps: float = 0.005,
) -> BandSpec:
    """Quantile bands with empirical bad rates and isotonic-smoothed semantics.

    Fixed-K quantile edges by default. With ``allow_merge=True``, adjacent
    bands whose empirical bad rates invert by more than ``merge_eps`` are
    merged until no material violation remains — trading band count for
    guaranteed-monotone empirical semantics.
    """
    F = np.asarray(latent, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if F.shape[0] != y_arr.shape[0]:
        raise ValueError("latent and y must have the same length")

    edges = _quantile_edges(F, n_bands, "quantile")

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
        },
    )
