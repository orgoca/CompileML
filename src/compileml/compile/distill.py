"""Teacher → whitebox distillation.

Any strong model (deep ensemble, neural network) can serve as the
teacher; what gets compiled is a small gradient-boosted whitebox trained
to reproduce the teacher's latent. Depth ≤ 2 keeps the artifact's
attribution *exact* (spec §7.3) — going deeper trades exactness for
fidelity and is warned about loudly.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor


def train_whitebox(
    X,
    teacher_latent,
    *,
    n_estimators: int = 30,
    max_depth: int = 2,
    learning_rate: float = 0.2,
    random_state: int = 42,
    loss: str = "squared_error",
) -> tuple[GradientBoostingRegressor, dict]:
    """Fit a whitebox GBM to a teacher's latent scores.

    Returns (model, metrics) where metrics quantifies distillation fidelity
    on the training data (pearson, spearman, mae, rmse, prediction range).
    """
    if max_depth > 2:
        warnings.warn(
            f"max_depth={max_depth} > 2: pairwise attribution will not be exact and "
            "the artifact will report a nonzero residual (see ARTIFACT_SPEC.md §7.3).",
            stacklevel=2,
        )

    X_arr = np.asarray(X, dtype=float)
    y = np.asarray(teacher_latent, dtype=float).reshape(-1)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        loss=loss,
    )
    model.fit(X_arr, y)

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
