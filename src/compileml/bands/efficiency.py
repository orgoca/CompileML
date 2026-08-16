"""Band efficiency: is the banding leaving discrimination on the table?

Bands quantize a continuous latent into a small ordered set. That always
discards *some* rank information — the question a policy owner needs
answered is **how much**, and **where**. Two complementary readings:

- **Gini gap** — continuous-latent Gini minus band-ordinal Gini. This is
  the portfolio-level discrimination the ladder discards; the headline
  "money on the table" number.
- **Within-band AUC** — can the latent still rank outcomes *inside* a
  band? ≈0.5 means the score is used up (splitting that band further buys
  nothing); materially above 0.5 means a finer cut there would separate
  risk the current ladder treats as homogeneous.

Both are estimates on a sample: small bands produce noisy AUCs, which is
why every within-band figure ships with a bootstrap confidence interval.
An upper CI bound near 0.5 is evidence of exhaustion; a lower bound well
above 0.5 is evidence of refinement headroom. Points in between are what
they look like — insufficient data to say.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from compileml.compile.quantize import rha


def _resolve_edges_int(bands, scale: int) -> tuple[np.ndarray, list[str], int]:
    """(edges_int, labels, scale) from an artifact, a BandSpec, or raw edges."""
    if isinstance(bands, dict) and "bands" in bands:  # full artifact
        block = bands["bands"]
        return (
            np.asarray(block["edges_int"], dtype=np.int64),
            [str(x) for x in block["labels"]],
            int(bands["scale"]),  # display scale lives at artifact top level
        )
    if hasattr(bands, "edges"):  # BandSpec
        edges = np.array([rha(float(e) * scale) for e in bands.edges], dtype=np.int64)
        return edges, list(bands.labels), scale
    edges_f = np.asarray(list(bands), dtype=float)
    edges = np.array([rha(float(e) * scale) for e in edges_f], dtype=np.int64)
    labels = [f"G{i + 1:02d}" for i in range(len(edges) - 1)]
    return edges, labels, scale


def _bootstrap_auc_ci(latent, y, n_boot: int, alpha: float, rng) -> tuple[float, float, float]:
    n = len(y)
    if n < 5 or len(np.unique(y)) < 2:
        return 0.5, 0.5, 0.5
    auc = float(roc_auc_score(y, latent))
    idx = np.arange(n)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        yb = y[take]
        boots[b] = 0.5 if len(np.unique(yb)) < 2 else float(roc_auc_score(yb, latent[take]))
    lo = float(np.clip(np.quantile(boots, alpha / 2), 0.0, 1.0))
    hi = float(np.clip(np.quantile(boots, 1 - alpha / 2), 0.0, 1.0))
    return auc, lo, hi


def band_efficiency(
    latent,
    y,
    bands,
    *,
    scale: int = 1000,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 7,
) -> dict:
    """Quantify the discrimination cost of a band ladder.

    Args:
        latent: Continuous latents in [0, 1] (whitebox predictions, clipped).
        y: Binary outcomes aligned with ``latent``.
        bands: A compiled artifact dict, a :class:`BandSpec`, or a sequence
            of float band edges. Assignment always runs on the fixed-point
            integer ladder — the deployed semantics.
        scale: Display scale used when ``bands`` is not an artifact.
        n_boot: Bootstrap resamples for the per-band AUC intervals.

    Returns:
        dict with ``continuous_gini``, ``band_ordinal_gini``, ``gini_gap``
        (the money on the table, in Gini points), ``gini_gap_pct``, a
        ``per_band`` table (n, bad_rate, within-band AUC + CI), and
        ``worst_band`` — the band whose lower CI bound sits highest above
        0.5, i.e. the strongest refinement candidate.
    """
    F = np.asarray(latent, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if F.shape[0] != y_arr.shape[0]:
        raise ValueError("latent and y must have the same length")

    edges_int, labels, scale = _resolve_edges_int(bands, scale)
    latent_int = np.array([rha(float(v) * scale) for v in F], dtype=np.int64)
    idx = np.clip(np.searchsorted(edges_int[1:-1], latent_int, side="right"), 0, len(edges_int) - 2)

    continuous_gini = 2 * float(roc_auc_score(y_arr, F)) - 1
    band_gini = 2 * float(roc_auc_score(y_arr, idx.astype(float))) - 1
    gap = continuous_gini - band_gini

    rng = np.random.default_rng(seed)
    per_band = []
    for b in range(len(labels)):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            per_band.append(
                {
                    "band": labels[b],
                    "n": 0,
                    "bad_rate": None,
                    "auc_within": None,
                    "auc_ci": None,
                    "verdict": "empty",
                }
            )
            continue
        auc, lo, hi = _bootstrap_auc_ci(F[mask], y_arr[mask], n_boot, alpha, rng)
        if hi <= 0.55:
            verdict = "exhausted"  # score is used up inside this band
        elif lo >= 0.55:
            verdict = "refinable"  # a finer cut here would separate real risk
        else:
            verdict = "inconclusive"  # CI spans both stories: need more data
        per_band.append(
            {
                "band": labels[b],
                "n": n,
                "bad_rate": float(y_arr[mask].mean()),
                "auc_within": round(auc, 4),
                "auc_ci": [round(lo, 4), round(hi, 4)],
                "verdict": verdict,
            }
        )

    candidates = [p for p in per_band if p["auc_ci"] is not None]
    worst = max(candidates, key=lambda p: p["auc_ci"][0], default=None)
    return {
        "continuous_gini": round(continuous_gini, 4),
        "band_ordinal_gini": round(band_gini, 4),
        "gini_gap": round(gap, 4),
        "gini_gap_pct": round(100 * gap / continuous_gini, 2) if continuous_gini > 0 else None,
        "per_band": per_band,
        "worst_band": worst["band"] if worst else None,
        "n_boot": int(n_boot),
    }
