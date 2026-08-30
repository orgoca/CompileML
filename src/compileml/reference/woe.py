"""A weight-of-evidence logistic regression, used as a *floor*.

Retention against a teacher is a one-sided number: it says how close the
compiled artifact came to a ceiling, and can never say that a
thirty-coefficient logistic regression would have beaten it. This module
supplies the other side of the comparison.

It is deliberately not a challenger-model toolkit. The binning is a shallow
supervised tree per feature, the encoding is textbook WOE, and the only
tuned hyper-parameter is the regularization strength. If a bank already has
a champion scorecard, its Gini is a better floor than anything fitted here
— every consumer of this module accepts a plain number in place of a fitted
model for exactly that reason.

The WOE tables are refit *inside* every cross-validation fold, so the
selected ``C`` is chosen against honestly out-of-fold encodings rather than
tables that have already seen the whole training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

C_GRID = (0.01, 0.1, 1.0, 10.0)


def _bin_index(thresholds: np.ndarray, column: np.ndarray) -> np.ndarray:
    """Bin membership under the codebase's ``x <= t`` convention.

    ``searchsorted(side="left")`` counts thresholds strictly below ``x``,
    which puts a value sitting exactly on an edge in the lower bin — the
    same rule ``build_scorecard`` uses via ``bisect_left``.
    """
    if thresholds.size == 0:
        return np.zeros(column.shape[0], dtype=int)
    return np.searchsorted(thresholds, column, side="left")


class _WOEEncoder(BaseEstimator, TransformerMixin):
    """Per-feature supervised binning, then weight-of-evidence values."""

    def __init__(self, max_bins: int = 8, min_samples_leaf: float = 0.02, random_state: int = 42):
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        n_features = X.shape[1]

        # Impute before binning so a NaN cannot silently become its own
        # split point; the artifact pipeline imputes at the baseline too.
        self.medians_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        X = self._impute(X)

        total_good = float(np.sum(y == 0))
        total_bad = float(np.sum(y == 1))
        if total_good == 0 or total_bad == 0:
            raise ValueError("reference model needs both classes present in y")

        self.thresholds_ = []
        self.woe_ = []
        self.iv_ = np.zeros(n_features, dtype=float)
        for j in range(n_features):
            column = X[:, j]
            tree = DecisionTreeClassifier(
                max_leaf_nodes=max(2, int(self.max_bins)),
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            edges: np.ndarray
            if np.unique(column).size < 2:
                edges = np.empty(0, dtype=float)  # constant feature: one bin
            else:
                tree.fit(column.reshape(-1, 1), y)
                inner = tree.tree_.feature != -2  # -2 is sklearn's leaf sentinel
                edges = np.unique(tree.tree_.threshold[inner]).astype(float)

            bins = _bin_index(edges, column)
            n_bins = edges.size + 1
            woe = np.zeros(n_bins, dtype=float)
            for b in range(n_bins):
                mask = bins == b
                # +0.5 keeps an empty or single-class bin finite.
                good = float(np.sum(y[mask] == 0)) + 0.5
                bad = float(np.sum(y[mask] == 1)) + 0.5
                dist_good = good / total_good
                dist_bad = bad / total_bad
                woe[b] = float(np.log(dist_good / dist_bad))
                self.iv_[j] += (dist_good - dist_bad) * woe[b]
            self.thresholds_.append(edges)
            self.woe_.append(woe)
        self.n_features_in_ = n_features
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        bad = ~np.isfinite(X)
        if bad.any():
            X = X.copy()
            X[bad] = np.take(self.medians_, np.nonzero(bad)[1])
        return X

    def transform(self, X):
        X = self._impute(np.asarray(X, dtype=float))
        out = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            out[:, j] = self.woe_[j][_bin_index(self.thresholds_[j], X[:, j])]
        return out


@dataclass
class ReferenceModel:
    """A fitted floor: WOE tables, a logistic layer, and its evidence."""

    kind: str
    feature_names: list[str]
    thresholds: list[list[float]]
    woe: list[list[float]]
    information_value: dict[str, float]
    coefficients: dict[str, float]
    intercept: float
    C: float
    cv_auc: float
    pipeline: object = field(repr=False, default=None)

    def score(self, X) -> np.ndarray:
        """Predicted probability of the positive class."""
        return np.asarray(self.pipeline.predict_proba(np.asarray(X, dtype=float))[:, 1])

    @property
    def n_coefficients(self) -> int:
        return len(self.coefficients)


def fit_reference(
    X,
    y,
    *,
    kind: str = "woe_logit",
    feature_names=None,
    max_bins: int = 8,
    min_samples_leaf: float = 0.02,
    cv: int = 3,
    C_grid=C_GRID,
    random_state: int = 42,
) -> ReferenceModel:
    """Fit the reference model a compiled artifact ought to beat.

    Args:
        X: Training rows.
        y: Binary outcomes.
        kind: Only ``"woe_logit"`` today — supervised binning,
            weight-of-evidence encoding, then logistic regression.
        max_bins: Maximum bins per feature (shallow by design).
        cv: Folds used to choose ``C``. The WOE tables are refit inside each
            fold, so the choice is not made against leaked encodings.

    Returns:
        A :class:`ReferenceModel` carrying the fitted pipeline plus the
        tables and information values that make it reviewable.
    """
    if kind != "woe_logit":
        raise ValueError(f"unknown reference kind {kind!r}; supported: 'woe_logit'")
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    names = (
        [str(n) for n in feature_names]
        if feature_names is not None
        else [f"f{i}" for i in range(X_arr.shape[1])]
    )
    if len(names) != X_arr.shape[1]:
        raise ValueError("feature_names length does not match X")

    pipe = Pipeline(
        [
            (
                "woe",
                _WOEEncoder(
                    max_bins=max_bins,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                ),
            ),
            ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    search = GridSearchCV(
        pipe, {"lr__C": list(C_grid)}, cv=cv, scoring="roc_auc", n_jobs=1, refit=True
    )
    search.fit(X_arr, y_arr)
    best = search.best_estimator_
    encoder, logit = best.named_steps["woe"], best.named_steps["lr"]

    return ReferenceModel(
        kind=kind,
        feature_names=names,
        thresholds=[[float(t) for t in ts] for ts in encoder.thresholds_],
        woe=[[float(w) for w in ws] for ws in encoder.woe_],
        information_value={n: float(v) for n, v in zip(names, encoder.iv_)},
        coefficients={n: float(c) for n, c in zip(names, logit.coef_.ravel())},
        intercept=float(logit.intercept_[0]),
        C=float(search.best_params_["lr__C"]),
        cv_auc=float(search.best_score_),
        pipeline=best,
    )


def reference_gini(reference, X, y) -> float:
    """Gini of a reference model (or a supplied Gini, passed through).

    Accepting a bare float is deliberate: a team with a champion scorecard
    has a better floor than anything :func:`fit_reference` produces, and
    every caller in CompileML takes either form.
    """
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        return float(reference)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    return 2.0 * float(roc_auc_score(y_arr, reference.score(X))) - 1.0
