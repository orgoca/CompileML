"""Score many rows through an artifact at once, with NumPy.

The runtime scores one row at a time in the standard library, which is what
lets it run anywhere. Selection scores forty compiled candidates on hundreds
of thousands of rows, where a Python loop costs minutes per candidate. This
module does the same arithmetic vectorised — the same integer sums, the same
float64 comparison, the same rounding — and the test suite asserts it agrees
with :func:`compileml.runtime.decide` on every row.

Learning side: it reads an artifact and never makes a decision anyone acts
on. Deployments use the runtime or an export.
"""

from __future__ import annotations

import numpy as np

from compileml.runtime.calibrate import PD_SCALE
from compileml.runtime.io import ArtifactError
from compileml.runtime.score import LEAF


def _div_rha_nonneg(num: np.ndarray, den) -> np.ndarray:
    """Spec §2.2 for non-negative numerators, in int64."""
    den = np.asarray(den, dtype=np.int64)
    return (2 * num + den) // (2 * den)


def _prepare(artifact: dict, X) -> np.ndarray:
    """The runtime's ``_prepare_row`` over a matrix (spec §4, §8)."""
    names = artifact["features"]["names"]
    X_arr = np.array(X, dtype=np.float64, copy=True)
    if X_arr.ndim != 2 or X_arr.shape[1] != len(names):
        raise ArtifactError(f"expected {len(names)} feature columns, got shape {X_arr.shape}")
    missing = np.isnan(X_arr)
    if missing.any():
        if artifact["features"].get("missing_policy", "baseline") == "reject":
            rows = int(missing.any(axis=1).sum())
            raise ArtifactError(f"{rows} row(s) have missing values and missing_policy is 'reject'")
        baseline = np.asarray(artifact["features"]["baseline"], dtype=np.float64)
        X_arr = np.where(missing, baseline[None, :], X_arr)
    if artifact["model"].get("input_precision", "float64") == "float32":
        X_arr = X_arr.astype(np.float32).astype(np.float64)
    return X_arr


def raw_micro_batch(model: dict, X: np.ndarray) -> np.ndarray:
    """``score_micro`` for every row: integer sums of the leaves each row reaches."""
    n = X.shape[0]
    acc = np.full(n, int(model["base_micro"]), dtype=np.int64)
    rows = np.arange(n)
    for tree in model["trees"]:
        feature = np.asarray(tree["feature"], dtype=np.int64)
        threshold = np.asarray(tree["threshold"], dtype=np.float64)
        left = np.asarray(tree["left"], dtype=np.int64)
        right = np.asarray(tree["right"], dtype=np.int64)
        value = np.asarray(tree["value_micro"], dtype=np.int64)
        node = np.zeros(n, dtype=np.int64)
        active = feature[node] != LEAF
        while active.any():
            idx = rows[active]
            at = node[idx]
            go_left = X[idx, feature[at]] <= threshold[at]
            node[idx] = np.where(go_left, left[at], right[at])
            active = feature[node] != LEAF
        acc += value[node]
    return acc


def calibrate_ppm_batch(
    latent_micro: np.ndarray, calibration: dict | None, micro_scale: int
) -> np.ndarray:
    """``calibrate_ppm`` for every row (spec §6)."""
    if not calibration:
        return _div_rha_nonneg(latent_micro * PD_SCALE, micro_scale)
    f = np.asarray(calibration["f_micro"], dtype=np.int64)
    pd = np.asarray(calibration["pd_ppm"], dtype=np.int64)
    k = len(f)
    out = np.empty_like(latent_micro)
    low = latent_micro <= f[0]
    high = latent_micro >= f[k - 1]
    out[low] = pd[0]
    out[high] = pd[k - 1]
    mid = ~(low | high)
    if mid.any():
        lm = latent_micro[mid]
        hi = np.searchsorted(f, lm, side="right")
        lo = hi - 1
        if calibration.get("mode", "linear_int") == "step":
            out[mid] = pd[lo]
        else:
            num = (pd[hi] - pd[lo]) * (lm - f[lo])
            out[mid] = pd[lo] + _div_rha_nonneg(num, f[hi] - f[lo])
    return out


def score_batch(artifact: dict, X) -> dict[str, np.ndarray]:
    """Score a matrix of rows: the runtime's score path, one array per output.

    Returns ``raw_micro``, ``latent_micro``, ``latent_int``, ``band_idx`` and
    ``pd_ppm`` as int64 arrays, plus ``pd`` as float64. Every integer equals
    what ``decide(artifact, row, explain=False)`` returns for that row.
    """
    model = artifact["model"]
    micro_scale = int(model["micro_scale"])
    ratio = micro_scale // int(artifact["scale"])
    X_arr = _prepare(artifact, X)

    raw = raw_micro_batch(model, X_arr)
    latent_micro = np.clip(raw, 0, micro_scale)
    latent_int = _div_rha_nonneg(latent_micro, ratio)
    edges = np.asarray(artifact["bands"]["edges_int"], dtype=np.int64)[1:-1]
    band_idx = np.searchsorted(edges, latent_int, side="right")
    pd_ppm = calibrate_ppm_batch(latent_micro, artifact.get("calibration"), micro_scale)
    return {
        "raw_micro": raw,
        "latent_micro": latent_micro,
        "latent_int": latent_int,
        "band_idx": band_idx,
        "pd_ppm": pd_ppm,
        "pd": pd_ppm / PD_SCALE,
    }
