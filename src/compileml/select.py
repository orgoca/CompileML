"""Select the target and the configuration on data the report never sees.

Three models with fixed jobs. The **ceiling** (the teacher) is the strongest
model the user can train under a declared budget; it prices compilation and
may supply soft targets, and it never ships. The **floor** — a WoE logistic
reference or a champion scorecard's Gini — says whether adopting the artifact
is worth it. The **candidate** is the depth-≤2 whitebox, one per
configuration, trained on labels, ceiling predictions, or a blend.

Three disjoint partitions. **Fit** trains everything; **Select** chooses the
target and the configuration; **Report** is read once, for the number that
gets published. The selected configuration is refit on Fit ∪ Select before
Report is read, so the selection curve ranks configurations, not the artifact
that ships.

Whether the whitebox should learn from the ceiling's probabilities or from
the labels is a parameter this module selects, not an assumption: on real
data, the whitebox trained on labels has beaten the distilled one. Every
step here is also a plain function, so the protocol can be followed by hand.

Learning side: NumPy and scikit-learn throughout. Nothing here runs at
decision time.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split

from compileml.artifact import build_artifact
from compileml.bands import monotone_quantile_bands
from compileml.batch import score_batch
from compileml.compile import extract_trees, train_whitebox
from compileml.compile.distill import refuse_missing_values
from compileml.reference import fit_reference
from compileml.runtime.io import canonical_hash

DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def default_grid() -> dict:
    """The default search space: five targets, four tree counts, two depths.

    ``monotone`` holds ``None`` and/or ``"declared"``; the declared set comes
    from ``compile_selected(monotone_constraints=...)`` and is never inferred.
    Depth 3 is not in the grid because it forfeits exact attribution; add it
    explicitly if that trade is acceptable.
    """
    return {
        "alpha": DEFAULT_ALPHAS,
        "trees": (20, 40, 80, 160),
        "depth": (1, 2),
        "monotone": (None,),
    }


# ------------------------------------------------------------------ metrics
def gini(y, score) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return 2.0 * float(roc_auc_score(y, np.asarray(score, dtype=float))) - 1.0


def ks_statistic(y, score) -> float:
    """Maximum separation between the score distributions of the two classes."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(score, kind="stable")
    pos = np.cumsum(y[order]) / y.sum()
    neg = np.cumsum(1 - y[order]) / (len(y) - y.sum())
    return float(np.max(np.abs(pos - neg)))


def brier(y, prob) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(prob, dtype=float) - y) ** 2))


# --------------------------------------------------------------- partitions
@dataclass
class Partitions:
    """Row indices of the three partitions, and how they were made."""

    fit: np.ndarray
    select: np.ndarray
    report: np.ndarray
    method: str

    @property
    def dev(self) -> np.ndarray:
        """Fit ∪ Select — what the selected configuration is refit on."""
        return np.concatenate([self.fit, self.select])


def make_partitions(
    y,
    *,
    method: str = "stratified",
    fractions=(0.6, 0.2, 0.2),
    date=None,
    group=None,
    explicit: dict | None = None,
    seed: int = 42,
) -> Partitions:
    """Split rows into Fit, Select and Report.

    - ``method="stratified"`` (default): stratified on the label.
    - ``date``: Report is the latest slice by date; Fit and Select are split
      within the earlier period.
    - ``group``: split by group so one applicant lands in one partition. Not
      stratified, which the method name records.
    - ``explicit``: ``{"fit": idx, "select": idx, "report": idx}``, checked
      for disjointness.
    """
    y = np.asarray(y).reshape(-1)
    n = len(y)
    if explicit is not None:
        parts = {k: np.asarray(explicit[k], dtype=int) for k in ("fit", "select", "report")}
        for k, idx in parts.items():
            if idx.size == 0 or idx.min() < 0 or idx.max() >= n or len(np.unique(idx)) != len(idx):
                raise ValueError(f"explicit {k} indices must be unique and within range")
        for a, b in (("fit", "select"), ("fit", "report"), ("select", "report")):
            if np.intersect1d(parts[a], parts[b]).size:
                raise ValueError(f"explicit partitions overlap: {a} and {b}")
        return Partitions(parts["fit"], parts["select"], parts["report"], "explicit")

    f_fit, f_select, f_report = (float(v) for v in fractions)
    if min(f_fit, f_select, f_report) <= 0 or abs(f_fit + f_select + f_report - 1.0) > 1e-9:
        raise ValueError("fractions must be positive and sum to 1")
    idx = np.arange(n)

    if date is not None:
        order = idx[np.argsort(np.asarray(date), kind="stable")]
        n_report = max(1, int(round(n * f_report)))
        report, rest = order[-n_report:], order[:-n_report]
        method = "out_of_time"
    else:
        if group is not None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=f_report, random_state=seed)
            rest_pos, report_pos = next(splitter.split(idx, groups=np.asarray(group)))
            rest, report = idx[rest_pos], idx[report_pos]
            method = "group"
        else:
            rest, report = train_test_split(idx, test_size=f_report, stratify=y, random_state=seed)
            method = "stratified"

    select_share = f_select / (f_fit + f_select)
    if group is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=select_share, random_state=seed + 1)
        fit_pos, select_pos = next(splitter.split(rest, groups=np.asarray(group)[rest]))
        fit, select = rest[fit_pos], rest[select_pos]
        method = method if method != "stratified" else "group"
        if date is not None:
            method = "out_of_time_group"
    else:
        fit, select = train_test_split(
            rest, test_size=select_share, stratify=y[rest], random_state=seed + 1
        )
    return Partitions(np.sort(fit), np.sort(select), np.sort(report), method)


def row_content_hashes(X, y) -> np.ndarray:
    """A 128-bit digest per row of its features and label, for disjointness checks."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y = np.asarray(y).reshape(-1)
    out = np.empty(len(y), dtype="S16")
    for i in range(len(y)):
        out[i] = hashlib.sha256(X[i].tobytes() + bytes([int(y[i])])).digest()[:16]
    return out


def partition_hashes(hashes: np.ndarray, parts: Partitions) -> dict:
    """One hex digest per partition, over its sorted row digests."""
    result = {}
    for name in ("fit", "select", "report"):
        rows = np.sort(hashes[getattr(parts, name)])
        result[name] = hashlib.sha256(b"".join(rows.tolist())).hexdigest()
    return result


def duplicate_rows(hashes: np.ndarray, parts: Partitions) -> int:
    """Rows in Select whose content appears in Fit, plus rows in Report whose
    content appears in Fit ∪ Select. Legitimate in credit data (coarse bins,
    repeat applicants), so a count rather than a failure."""
    fit_set = set(hashes[parts.fit].tolist())
    dup = sum(1 for h in hashes[parts.select].tolist() if h in fit_set)
    dev_set = fit_set | set(hashes[parts.select].tolist())
    dup += sum(1 for h in hashes[parts.report].tolist() if h in dev_set)
    return int(dup)


# ------------------------------------------------------------------ ceiling
def ceiling_predict(model, X) -> np.ndarray:
    """A ceiling's score: positive-class probability, or the prediction."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1].astype(float)
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def freeze_ceiling(model):
    """The configuration to refit: a search's best estimator, or the model itself."""
    frozen = getattr(model, "best_estimator_", model)
    try:
        clone(frozen)
    except Exception as exc:  # pragma: no cover - depends on the user's object
        raise TypeError(
            "the ceiling factory must return a scikit-learn-compatible estimator "
            "(or a fitted search exposing best_estimator_) so its frozen "
            "configuration can be refit for out-of-fold predictions; otherwise pass "
            "ceiling_oof= and ceiling_scores= instead"
        ) from exc
    return frozen


def params_hash(estimator) -> str:
    """A digest of the frozen hyperparameters, so provenance can name what was refit."""
    try:
        text = repr(sorted(estimator.get_params(deep=False).items()))
    except Exception:  # pragma: no cover
        text = repr(estimator)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class _Ceiling:
    kind: str  # "factory" | "predictions"
    factory: object = None
    frozen: object = None
    oof: np.ndarray | None = None  # full-length, for kind == "predictions"
    scores: np.ndarray | None = None
    budget: dict | None = None
    factory_calls: int = 0
    fits: int = 0

    def tune(self, X, y, idx):
        """Call the factory once on ``idx`` and freeze what it returns."""
        self.frozen = freeze_ceiling(self.factory(X[idx], y[idx]))
        self.factory_calls += 1
        self.fits += 1

    def out_of_fold(self, X, y, idx, k: int, seed: int) -> np.ndarray:
        """Cross-fitted predictions on ``idx``: k refits of the frozen configuration."""
        if self.kind == "predictions":
            return self.oof[idx]
        out = np.empty(len(idx), dtype=float)
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for train_pos, held_pos in folds.split(idx, y[idx]):
            model = clone(self.frozen).fit(X[idx[train_pos]], y[idx[train_pos]])
            self.fits += 1
            out[held_pos] = ceiling_predict(model, X[idx[held_pos]])
        return out

    def score(self, X, y, train_idx, eval_idx) -> np.ndarray:
        """The frozen configuration refit on ``train_idx``, scored on ``eval_idx``."""
        if self.kind == "predictions":
            return self.scores[eval_idx]
        model = clone(self.frozen).fit(X[train_idx], y[train_idx])
        self.fits += 1
        return ceiling_predict(model, X[eval_idx])

    @property
    def family(self) -> str:
        if self.kind == "predictions":
            return "user_predictions"
        cls = type(self.frozen)
        return f"{cls.__module__}.{cls.__name__}"


# --------------------------------------------------------------------- grid
def _resolve_grid(grid, monotone_constraints, has_ceiling) -> tuple[dict, list[str]]:
    g = dict(default_grid())
    g.update(grid or {})
    notes = []
    alphas = tuple(float(a) for a in g["alpha"])
    if not has_ceiling:
        if any(a < 1.0 for a in alphas):
            notes.append("no ceiling predictions: alpha forced to (1.0,) — labels only")
        alphas = (1.0,)
    else:
        for endpoint in (0.0, 1.0):
            if endpoint not in alphas:
                alphas = tuple(sorted({*alphas, endpoint}))
                notes.append(f"alpha endpoint {endpoint} added: endpoints are always searched")
    if any(not 0.0 <= a <= 1.0 for a in alphas):
        raise ValueError("alpha values must lie in [0, 1]")
    trees = tuple(sorted({int(t) for t in g["trees"]}))
    depths = tuple(sorted({int(d) for d in g["depth"]}))
    if any(d > 2 for d in depths):
        notes.append("depth > 2 in the grid: those candidates forfeit exact attribution")
    monos = []
    for m in g["monotone"]:
        if m is None:
            monos.append(None)
        elif m == "declared":
            if monotone_constraints is None:
                raise ValueError("grid asks for the declared monotone set but none was given")
            monos.append(monotone_constraints)
        else:
            raise ValueError("grid['monotone'] entries must be None or 'declared'")
    return {"alpha": alphas, "trees": trees, "depth": depths, "monotone": tuple(monos)}, notes


def _config_key(alpha, trees, depth, mono_index) -> tuple:
    return (float(alpha), int(trees), int(depth), int(mono_index))


def evaluate_grid(
    X,
    y,
    train_idx,
    eval_idx,
    *,
    grid: dict,
    oof_train: np.ndarray | None,
    feature_names,
    n_bands: int,
    backend: str,
    learning_rate: float,
    sample_weight=None,
    seed: int = 42,
) -> dict[tuple, dict]:
    """Every configuration in the grid, compiled on ``train_idx`` and scored on ``eval_idx``.

    One fit per (alpha, depth, monotone) at the largest tree count; the
    smaller counts are prefixes of that fit, extracted once and truncated.
    Each prefix is quantized, calibrated and banded on the training rows and
    scored on the evaluation rows through the batch scorer — the compiled
    artifact is what is ranked, not the float whitebox.
    """
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_ev = X[eval_idx]
    w_tr = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[train_idx]
    baseline = np.median(X_tr, axis=0)
    positives = int(y_tr.sum())
    trees_wanted = set(grid["trees"])
    max_trees = max(grid["trees"])
    rows: dict[tuple, dict] = {}

    for alpha in grid["alpha"]:
        target = y_tr.astype(float) if alpha >= 1.0 else alpha * y_tr + (1.0 - alpha) * oof_train
        for depth in grid["depth"]:
            for m_index, mono in enumerate(grid["monotone"]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model, _ = train_whitebox(
                        X_tr,
                        target,
                        n_estimators=max_trees,
                        max_depth=depth,
                        learning_rate=learning_rate,
                        random_state=seed,
                        monotone_constraints=mono,
                        sample_weight=w_tr,
                        backend=backend,
                    )
                    extracted = extract_trees(model)
                    stages = {}
                    for i, pred in enumerate(model.staged_predict(X_tr), start=1):
                        if i in trees_wanted:
                            stages[i] = np.clip(pred, 0.0, 1.0)
                    for k in grid["trees"]:
                        prefix = replace(extracted, trees=extracted.trees[:k])
                        latent = stages[k]
                        artifact = build_artifact(
                            prefix,
                            feature_names,
                            baseline,
                            monotone_quantile_bands(latent, y_tr, n_bands=n_bands),
                            calibration_latent=latent,
                            calibration_y=y_tr,
                            monotone_constraints=mono,
                        )
                        scored = score_batch(artifact, X_ev)
                        rows[_config_key(alpha, k, depth, m_index)] = {
                            "alpha": float(alpha),
                            "n_estimators": int(k),
                            "max_depth": int(depth),
                            "monotone": mono is not None,
                            "eps": positives / (k * (2**depth - 1)),
                            "_raw": scored["raw_micro"],
                            "_pd": scored["pd"],
                        }
    return rows


# ---------------------------------------------------------------- selection
def _metric(name: str, y, raw, pd) -> float:
    if name == "gini":
        return gini(y, raw)
    if name == "brier":
        return -brier(y, pd)  # higher is better throughout selection
    raise ValueError("selection_metric must be 'gini' or 'brier'")


def choose(
    rows: dict[tuple, dict], y_eval, *, metric: str, tie_se: float, n_boot: int, seed: int
) -> dict:
    """One-SE-simplest: the simplest configuration statistically tied with the best.

    Bootstrap the evaluation rows to get the standard error of the best
    configuration's metric; everything within ``tie_se`` standard errors is
    tied. Among the tied, prefer α = 1 (no training dependency on the
    ceiling), then fewer trees, then lower depth. The α preference is a
    governance prior, recorded as the tie rule, not an empirical claim.
    """
    y_eval = np.asarray(y_eval, dtype=int)
    for row in rows.values():
        row["gini"] = gini(y_eval, row["_raw"])
        row["brier"] = brier(y_eval, row["_pd"])
        row["score"] = _metric(metric, y_eval, row["_raw"], row["_pd"])
    best_key = max(rows, key=lambda k: rows[k]["score"])
    best = rows[best_key]

    rng = np.random.default_rng(seed)
    n = len(y_eval)
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        samples.append(_metric(metric, y_eval[pick], best["_raw"][pick], best["_pd"][pick]))
    se = float(np.std(samples, ddof=1)) if n_boot > 1 else 0.0

    tied = [k for k, r in rows.items() if r["score"] >= best["score"] - tie_se * se]
    chosen_key = min(tied, key=lambda k: (-k[0], k[1], k[2], k[3]))
    for k, r in rows.items():
        r["tied"] = k in tied
        r["selected"] = k == chosen_key
    return {
        "key": chosen_key,
        "best_key": best_key,
        "se": se,
        "n_tied": len(tied),
        "metric": metric,
        "tie_se": tie_se,
        "n_boot": n_boot,
    }


# ------------------------------------------------------------------- result
@dataclass
class SelectionResult:
    """What ``compile_selected`` returns: the artifact, the curve, and one report."""

    artifact: dict | None
    selection_curve: list[dict]
    selected: dict
    partitions: Partitions
    floor_gate: dict
    notes: list[str]
    report_evaluations: int = 0
    _state: dict = field(default_factory=dict, repr=False)

    def report(self) -> dict:
        """Evaluate the shipped artifact on Report, once.

        Gini, KS and Brier for artifact, ceiling and floor; retention against
        the ceiling and the ratio against the floor at equal weight, each with
        a bootstrap interval. The figures are written into the artifact's
        provenance and the artifact is rehashed, so what ships carries its own
        reported cost. A second call raises: nothing may be chosen after
        Report is read, and this counter is hygiene rather than proof.
        """
        if self.report_evaluations >= 1:
            raise RuntimeError(
                "Report has already been evaluated once; a second evaluation after "
                "changing anything would invalidate the published figure. Rerun the "
                "whole pipeline if the configuration changes."
            )
        if self.artifact is None:
            raise RuntimeError("no artifact: the selected configuration did not clear the floor")
        s = self._state
        y_rep = s["y"][self.partitions.report]
        scored = score_batch(self.artifact, s["X"][self.partitions.report])
        art_raw, art_pd = scored["raw_micro"], scored["pd"]
        ceil = s["ceiling_report"]
        floor_scores, floor_gini = s["floor_report"]

        def block(raw, prob):
            return {
                "gini": gini(y_rep, raw),
                "ks": ks_statistic(y_rep, raw),
                "brier": brier(y_rep, prob) if prob is not None else None,
            }

        art_m = block(art_raw, art_pd)
        ceil_m = (
            block(ceil, ceil) if ceil is not None else {"gini": None, "ks": None, "brier": None}
        )
        floor_m = (
            block(floor_scores, floor_scores)
            if floor_scores is not None
            else {"gini": floor_gini, "ks": None, "brier": None}
        )

        rng = np.random.default_rng(s["seed"] + 7)
        n = len(y_rep)
        ret, ratio, art_g = [], [], []
        for _ in range(s["n_boot"]):
            pick = rng.integers(0, n, n)
            g_art = gini(y_rep[pick], art_raw[pick])
            art_g.append(g_art)
            if ceil is not None:
                g_ceil = gini(y_rep[pick], ceil[pick])
                ret.append(100 * g_art / g_ceil if g_ceil > 0 else float("nan"))
            if floor_scores is not None:
                g_floor = gini(y_rep[pick], floor_scores[pick])
                ratio.append(100 * g_art / g_floor if g_floor > 0 else float("nan"))
            elif floor_gini:
                ratio.append(100 * g_art / floor_gini)

        def interval(values):
            values = [v for v in values if v == v]
            if not values:
                return None
            lo, hi = np.percentile(values, [2.5, 97.5])
            return [float(lo), float(hi)]

        report = {
            "rows": int(n),
            "artifact": art_m,
            "ceiling": ceil_m,
            "floor": floor_m,
            "retention_pct": (
                100 * art_m["gini"] / ceil_m["gini"]
                if ceil_m["gini"] and ceil_m["gini"] > 0
                else None
            ),
            "retention_ci": interval(ret),
            "floor_ratio_pct": (
                100 * art_m["gini"] / floor_m["gini"]
                if floor_m["gini"] and floor_m["gini"] > 0
                else None
            ),
            "floor_ratio_ci": interval(ratio),
            "artifact_gini_ci": interval(art_g),
            "n_boot": int(s["n_boot"]),
            "evaluations": 1,
        }
        self.report_evaluations = 1
        prov = self.artifact["metadata"]["provenance"]
        prov["report"] = _plain(report)
        self.artifact["artifact_hash"] = canonical_hash(self.artifact)
        report["artifact_hash"] = self.artifact["artifact_hash"]
        return report


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _column(X_raw, spec, name):
    if spec is None:
        return None
    if isinstance(spec, str):
        if not hasattr(X_raw, "columns"):
            raise ValueError(f"{name} is a column name but X has no columns")
        return np.asarray(X_raw[spec])
    return np.asarray(spec).reshape(-1)


# ---------------------------------------------------------- the entry point
def compile_selected(
    X,
    y,
    *,
    ceiling=None,
    ceiling_oof=None,
    ceiling_scores=None,
    ceiling_budget: dict | None = None,
    reference="woe",
    partitions="stratified",
    date=None,
    group=None,
    row_id=None,
    grid: dict | None = None,
    monotone_constraints=None,
    selection_metric: str = "gini",
    tie_se: float = 1.0,
    n_boot: int = 200,
    k_folds: int = 5,
    min_select_events: int = 300,
    n_bands: int = 10,
    feature_names=None,
    reasons: dict | None = None,
    sample_weight=None,
    allow_below_floor: bool = False,
    eps_warn: float = 10.0,
    duplicate_threshold: float = 0.01,
    backend: str = "hist",
    learning_rate: float = 0.2,
    seed: int = 42,
    build_kwargs: dict | None = None,
) -> SelectionResult:
    """Compile the whitebox whose target and configuration were chosen on Select.

    Args:
        X, y: All rows and binary outcomes; the function partitions them.
        ceiling: A callable ``(X_fit, y_fit) -> fitted model``. It is called
            once, its configuration is frozen, and that configuration is
            refit ``k_folds`` times for cross-fitted soft targets. One search
            plus K fits — never K searches.
        ceiling_oof, ceiling_scores: For users who can only supply
            predictions: out-of-fold predictions used as soft targets, and
            the ceiling's scores used for retention, both full-length.
            Provenance then records ``cross_fitted: "user_asserted"``.
        ceiling_budget: Free-form dict recorded in provenance — trials, CV
            scheme, search space — so retention is comparable across artifacts.
        reference: ``"woe"`` fits the WoE logistic floor inside Fit; a float
            is a champion scorecard's Gini; ``None`` skips the floor gate,
            with a warning.
        partitions: ``"stratified"``, or an explicit dict of index arrays.
        date, group, row_id: Arrays (or column names, when X has columns).
            ``date`` makes Report the latest slice; ``group`` keeps one
            applicant in one partition; ``row_id`` turns partition overlap
            into a hard failure.
        grid: See :func:`default_grid`.
        selection_metric: ``"gini"`` (default) or ``"brier"``.
        tie_se: Width of the tie band in standard errors.
        min_select_events: Below this many events in Select, selection uses
            nested cross-validation over Fit ∪ Select instead, with a warning.
        allow_below_floor: Build the artifact even if the selected
            configuration does not beat the floor on Select.
        eps_warn: Provisional events-per-split threshold below which the
            grid should hold interior α values; a diagnostic, never an input.
        duplicate_threshold: Maximum share of Select and Report rows whose
            content also appears in an earlier partition.
        backend: ``"hist"`` (the fast histogram backend) or ``"gbr"``.

    Returns:
        A :class:`SelectionResult`. Call ``.report()`` once for the figure.
    """
    X_raw = X
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2-D array of rows")
    y_arr = np.asarray(y).reshape(-1)
    if len(y_arr) != X_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if not np.isin(y_arr, [0, 1]).all():
        raise ValueError("y must contain only 0 and 1")
    y_arr = y_arr.astype(int)
    names = (
        [str(n) for n in feature_names]
        if feature_names is not None
        else (
            [str(c) for c in X_raw.columns]
            if hasattr(X_raw, "columns")
            else [f"f{i}" for i in range(X_arr.shape[1])]
        )
    )
    if len(names) != X_arr.shape[1]:
        raise ValueError("feature_names length does not match X")
    refuse_missing_values(X_arr, names)  # before the ceiling spends its budget
    notes: list[str] = []

    # ---------------------------------------------------------- partitions
    date_arr, group_arr, row_ids = (
        _column(X_raw, v, n) for v, n in ((date, "date"), (group, "group"), (row_id, "row_id"))
    )
    if isinstance(partitions, dict):
        parts = make_partitions(y_arr, explicit=partitions, seed=seed)
    elif partitions == "stratified":
        parts = make_partitions(y_arr, date=date_arr, group=group_arr, seed=seed)
    else:
        raise ValueError("partitions must be 'stratified' or a dict of index arrays")
    if row_ids is not None:
        overlap = 0
        for a, b in (
            (parts.fit, parts.select),
            (parts.fit, parts.report),
            (parts.select, parts.report),
        ):
            overlap += int(np.intersect1d(row_ids[a], row_ids[b]).size)
        if overlap:
            raise ValueError(f"{overlap} row id(s) appear in more than one partition")
    hashes = row_content_hashes(X_arr, y_arr)
    duplicates = duplicate_rows(hashes, parts)
    dup_fraction = duplicates / (len(parts.select) + len(parts.report))
    if dup_fraction > duplicate_threshold:
        raise ValueError(
            f"{duplicates} rows in Select or Report ({dup_fraction:.1%}) have content "
            "identical to a "
            f"row in an earlier partition, above duplicate_threshold={duplicate_threshold}. "
            "If the same entity appears more than once, pass group= so each lands in one "
            "partition. If different entities share identical rows by construction — a "
            "no-record code in every feature, say — nothing leaks: raise duplicate_threshold "
            f"above {dup_fraction:.3f}."
        )

    # ------------------------------------------------------------- ceiling
    if ceiling is not None and (ceiling_oof is not None or ceiling_scores is not None):
        raise ValueError(
            "pass either ceiling= (a factory) or ceiling_oof=/ceiling_scores=, not both"
        )
    if ceiling is not None:
        ceil = _Ceiling(kind="factory", factory=ceiling, budget=ceiling_budget)
    elif ceiling_oof is not None:
        oof = np.asarray(ceiling_oof, dtype=float).reshape(-1)
        scores = np.asarray(
            ceiling_scores if ceiling_scores is not None else ceiling_oof, dtype=float
        ).reshape(-1)
        if len(oof) != len(y_arr) or len(scores) != len(y_arr):
            raise ValueError("ceiling_oof and ceiling_scores must be full-length")
        ceil = _Ceiling(kind="predictions", oof=oof, scores=scores, budget=ceiling_budget)
    else:
        ceil = None
        notes.append("no ceiling: retention cannot be reported; the floor is the only yardstick")

    resolved, grid_notes = _resolve_grid(grid, monotone_constraints, ceil is not None)
    notes.extend(grid_notes)

    # --------------------------------------------------------------- floor
    if reference is None:
        warnings.warn(
            "no reference floor: the artifact cannot be compared with what a risk team "
            "builds in an afternoon. Pass reference='woe' or a champion scorecard's Gini.",
            stacklevel=2,
        )
    floor_is_gini = isinstance(reference, (int, float)) and not isinstance(reference, bool)

    # ----------------------------------------------------------- selection
    select_events = int(y_arr[parts.select].sum())
    nested = select_events < min_select_events
    if nested:
        warnings.warn(
            f"Select holds {select_events} events, below min_select_events={min_select_events}: "
            "selecting with nested cross-validation over Fit ∪ Select instead. Report stays "
            "held out.",
            stacklevel=2,
        )
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)

    def run_split(train_idx, eval_idx):
        oof_train = None
        if ceil is not None and any(a < 1.0 for a in resolved["alpha"]):
            if ceil.kind == "factory" and (ceil.frozen is None or nested):
                ceil.tune(X_arr, y_arr, train_idx)
            oof_train = ceil.out_of_fold(X_arr, y_arr, train_idx, k_folds, seed)
        elif ceil is not None and ceil.kind == "factory" and ceil.frozen is None:
            ceil.tune(X_arr, y_arr, train_idx)
        rows = evaluate_grid(
            X_arr,
            y_arr,
            train_idx,
            eval_idx,
            grid=resolved,
            oof_train=oof_train,
            feature_names=names,
            n_bands=n_bands,
            backend=backend,
            learning_rate=learning_rate,
            sample_weight=sw,
            seed=seed,
        )
        floor_scores = None
        if reference is not None and not floor_is_gini:
            ref = fit_reference(
                X_arr[train_idx], y_arr[train_idx], feature_names=names, random_state=seed
            )
            floor_scores = ref.score(X_arr[eval_idx])
        return rows, floor_scores

    if not nested:
        rows, floor_select = run_split(parts.fit, parts.select)
        y_eval = y_arr[parts.select]
    else:
        dev = parts.dev
        folds = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        pooled: dict[tuple, dict] = {}
        floor_parts, y_parts = [], []
        for train_pos, held_pos in folds.split(dev, y_arr[dev]):
            fold_rows, fold_floor = run_split(dev[train_pos], dev[held_pos])
            y_parts.append(y_arr[dev[held_pos]])
            if fold_floor is not None:
                floor_parts.append(fold_floor)
            for key, row in fold_rows.items():
                agg = pooled.setdefault(
                    key,
                    {
                        **{k: v for k, v in row.items() if not k.startswith("_")},
                        "_raw": [],
                        "_pd": [],
                    },
                )
                agg["_raw"].append(row["_raw"])
                agg["_pd"].append(row["_pd"])
        rows = {
            k: {**r, "_raw": np.concatenate(r["_raw"]), "_pd": np.concatenate(r["_pd"])}
            for k, r in pooled.items()
        }
        y_eval = np.concatenate(y_parts)
        floor_select = np.concatenate(floor_parts) if floor_parts else None

    selected = choose(
        rows, y_eval, metric=selection_metric, tie_se=tie_se, n_boot=n_boot, seed=seed
    )
    chosen = rows[selected["key"]]
    eps_min = min(r["eps"] for r in rows.values())
    if eps_min < eps_warn and set(resolved["alpha"]) <= {0.0, 1.0} and ceil is not None:
        warnings.warn(
            f"minimum events per split in the grid is {eps_min:.1f}, below eps_warn={eps_warn}, "
            "and the "
            "alpha grid holds only endpoints; interior values (a blend of labels and ceiling "
            "predictions) are worth searching at this event count. The threshold is provisional.",
            stacklevel=2,
        )

    # ---------------------------------------------------------- floor gate
    if reference is None:
        floor_gini_select = None
    elif floor_is_gini:
        floor_gini_select = float(reference)
    else:
        floor_gini_select = gini(y_eval, floor_select)
    clears = floor_gini_select is None or chosen["gini"] >= floor_gini_select
    floor_gate = {
        "passed": bool(clears),
        "select_gini": chosen["gini"],
        "floor_gini": floor_gini_select,
        "enforced": bool(reference is not None and not allow_below_floor),
    }

    curve = sorted(
        ({k: v for k, v in r.items() if not k.startswith("_")} for r in rows.values()),
        key=lambda r: (-r["score"], r["n_estimators"], r["max_depth"]),
    )
    for r in curve:
        r.pop("score", None)

    result_common = dict(
        selection_curve=curve,
        selected={
            "alpha": chosen["alpha"],
            "n_estimators": chosen["n_estimators"],
            "max_depth": chosen["max_depth"],
            "monotone": chosen["monotone"],
            "select_gini": chosen["gini"],
            "select_brier": chosen["brier"],
            "eps": chosen["eps"],
            **{k: v for k, v in selected.items() if k not in ("key", "best_key")},
            "nested": nested,
        },
        partitions=parts,
        floor_gate=floor_gate,
        notes=notes,
    )
    if not clears and not allow_below_floor:
        return SelectionResult(artifact=None, **result_common)

    # --------------------------------------------------------- final refit
    dev = parts.dev
    alpha = chosen["alpha"]
    mono = monotone_constraints if chosen["monotone"] else None
    cross_fitted: object = None
    if alpha < 1.0:
        if ceil.kind == "factory" and (ceil.frozen is None or nested):
            ceil.tune(X_arr, y_arr, dev)
        oof_dev = ceil.out_of_fold(X_arr, y_arr, dev, k_folds, seed)
        target = alpha * y_arr[dev] + (1.0 - alpha) * oof_dev
        cross_fitted = True if ceil.kind == "factory" else "user_asserted"
    else:
        target = y_arr[dev].astype(float)
        if ceil is not None and ceil.kind == "factory" and (ceil.frozen is None or nested):
            ceil.tune(X_arr, y_arr, dev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = train_whitebox(
            X_arr[dev],
            target,
            n_estimators=chosen["n_estimators"],
            max_depth=chosen["max_depth"],
            learning_rate=learning_rate,
            random_state=seed,
            monotone_constraints=mono,
            sample_weight=None if sw is None else sw[dev],
            backend=backend,
        )
    latent = np.clip(model.predict(X_arr[dev]), 0.0, 1.0)
    part_hashes = partition_hashes(hashes, parts)
    provenance = {
        "target": {"alpha": alpha, "uses_ceiling": alpha < 1.0, "loss": "squared_error"},
        "selection": {
            "metric": selection_metric,
            "rule": "one_se_simplest",
            "tie_se": tie_se,
            "se": selected["se"],
            "n_tied": selected["n_tied"],
            "n_configs": len(rows),
            "n_boot": n_boot,
            "nested": nested,
            "k_folds": k_folds if nested else None,
            "tie_rule": "prefer alpha=1, then fewer trees, then lower depth — a governance prior",
        },
        "partitions": {
            "method": parts.method + ("_nested" if nested else ""),
            "fit_rows": int(len(parts.fit)),
            "select_rows": int(len(parts.select)),
            "report_rows": int(len(parts.report)),
            "content_hashes": part_hashes,
            "duplicate_rows": duplicates,
            "duplicate_fraction": dup_fraction,
            "duplicate_threshold": duplicate_threshold,
            "row_id_overlap": 0 if row_ids is not None else None,
        },
        "soft_targets": {"cross_fitted": cross_fitted, "k": k_folds if alpha < 1.0 else None},
        "ceiling": (
            None
            if ceil is None
            else {
                "kind": ceil.kind,
                "family": ceil.family,
                "params_hash": params_hash(ceil.frozen) if ceil.kind == "factory" else None,
                "budget": ceil.budget,
                "factory_calls": ceil.factory_calls,
                "fits": ceil.fits,
            }
        ),
        "floor": (
            None
            if reference is None
            else {
                "kind": "gini" if floor_is_gini else "woe_logit",
                "select_gini": floor_gini_select,
            }
        ),
        "eps_min": eps_min,
        "eps_warn": eps_warn,
        "report": None,
    }
    build_kwargs = dict(build_kwargs or {})
    metadata = {**build_kwargs.pop("metadata", {}), "provenance": _plain(provenance)}
    artifact = build_artifact(
        model,
        names,
        np.median(X_arr[dev], axis=0),
        monotone_quantile_bands(latent, y_arr[dev], n_bands=n_bands),
        calibration_latent=latent,
        calibration_y=y_arr[dev],
        monotone_constraints=mono,
        reasons=reasons,
        metadata=metadata,
        X_sample=X_arr[dev][:500],
        **build_kwargs,
    )

    ceiling_report = ceil.score(X_arr, y_arr, dev, parts.report) if ceil is not None else None
    if reference is None:
        floor_report = (None, None)
    elif floor_is_gini:
        floor_report = (None, float(reference))
    else:
        ref = fit_reference(X_arr[dev], y_arr[dev], feature_names=names, random_state=seed)
        floor_report = (ref.score(X_arr[parts.report]), None)
    if ceil is not None:
        artifact["metadata"]["provenance"]["ceiling"]["fits"] = ceil.fits
        artifact["artifact_hash"] = canonical_hash(artifact)

    return SelectionResult(
        artifact=artifact,
        _state={
            "X": X_arr,
            "y": y_arr,
            "ceiling_report": ceiling_report,
            "floor_report": floor_report,
            "n_boot": n_boot,
            "seed": seed,
        },
        **result_common,
    )
