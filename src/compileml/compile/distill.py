"""Teacher → whitebox distillation.

Any strong model (deep ensemble, neural network) can serve as the
teacher; what gets compiled is a small gradient-boosted whitebox trained
to reproduce the teacher's latent. Depth ≤ 2 keeps the artifact's
attribution *exact* (spec §7.3) — going deeper trades exactness for
fidelity and is warned about loudly.

With ``monotone_constraints`` the whitebox is trained with
``HistGradientBoostingRegressor`` (the only sklearn GBM that enforces
``monotonic_cst`` during growth); without them the classic
``GradientBoostingRegressor`` path is untouched, byte for byte.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from compileml.compile.monotone import normalize_constraints


def train_whitebox(
    X,
    target=None,
    *,
    n_estimators: int = 30,
    max_depth: int = 2,
    learning_rate: float = 0.2,
    random_state: int = 42,
    loss: str = "squared_error",
    monotone_constraints=None,
    teacher_latent=None,
    sample_weight=None,
):
    """Fit a whitebox GBM to a target.

    ``target`` is any per-row vector to regress onto: a teacher's latent
    probabilities (distillation), the binary labels themselves, or a blend
    of the two. Distillation is a choice here, not an assumption — at a
    depth-2 budget, fitting a soft target can spend capacity on teacher
    noise instead of outcome, so it is worth sweeping rather than assuming
    (``sweep_whitebox(alpha_grid=...)``).

    Returns (model, metrics) where metrics quantifies fidelity to the target
    on the training data (pearson, spearman, mae, rmse, prediction range).

    ``monotone_constraints`` takes a per-feature sequence of -1/0/+1 or a
    dict keyed by feature index (or by name, when ``X`` is a DataFrame
    carrying column names). Any nonzero sign switches
    the backend to ``HistGradientBoostingRegressor``; ``None`` (or all
    zeros) keeps the classic ``GradientBoostingRegressor``.

    ``sample_weight`` — one non-negative weight per row — tells the fit where
    fidelity matters. A whitebox has a fixed budget and, unweighted, spends
    it where most of the squared error is: the largest segment and the busiest
    part of the score range. Up-weighting a segment, or the rows near its
    cutoff range, moves that budget; other segments pay for it, so re-check
    them with :func:`~compileml.tune.retention_by_segment`. Weighting changes
    how the model ranks, not the PD: calibration is fitted afterwards on
    outcomes. The returned fidelity metrics stay unweighted. Weighting cannot
    create an effect depth 2 cannot express — a segment-only interaction is
    three-way — see the tuning guide.
    """
    if teacher_latent is not None:
        if target is not None:
            raise TypeError("pass target or teacher_latent, not both")
        warnings.warn(
            "teacher_latent= is deprecated and will be removed in a future release; "
            "the parameter is now called target=, since labels and label/teacher "
            "blends are equally valid targets.",
            DeprecationWarning,
            stacklevel=2,
        )
        target = teacher_latent
    if target is None:
        raise TypeError("train_whitebox() missing required argument: 'target'")

    if max_depth > 2:
        warnings.warn(
            f"max_depth={max_depth} > 2: pairwise attribution will not be exact and "
            "the artifact will report a nonzero residual (see ARTIFACT_SPEC.md §7.3).",
            stacklevel=2,
        )

    X_arr = np.asarray(X, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weights.shape[0] != y.shape[0]:
            raise ValueError("sample_weight must have one weight per row")
        if (weights < 0).any() or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("sample_weight must be finite, non-negative, and not all zero")
    feature_names = list(X.columns) if hasattr(X, "columns") else None
    cst = normalize_constraints(monotone_constraints, X_arr.shape[1], feature_names=feature_names)
    if cst is not None:
        model = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            monotonic_cst=cst,
            early_stopping=False,
            max_leaf_nodes=None,
            random_state=random_state,
            loss=loss,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            loss=loss,
        )
    model.fit(X_arr, y, sample_weight=weights)

    y_hat = np.clip(model.predict(X_arr), 0.0, 1.0)
    metrics = {
        "pearson": float(pearsonr(y, y_hat)[0]),
        "spearman": float(spearmanr(y, y_hat)[0]),
        "mae": float(np.mean(np.abs(y - y_hat))),
        "rmse": float(np.sqrt(np.mean((y - y_hat) ** 2))),
        "min_pred": float(np.min(y_hat)),
        "max_pred": float(np.max(y_hat)),
    }
    return model, metrics
