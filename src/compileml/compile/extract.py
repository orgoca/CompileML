"""Tree extraction from fitted models into portable float arrays.

Supported families: sklearn gradient boosting (classic and hist),
XGBoost, LightGBM.
Extraction produces float trees plus (base, learning_rate); quantization
to the integer artifact model happens afterwards in ``quantize.py``.

XGBoost note: hist trees compare inputs in float32 with strict ``<``.
We convert to the scorer's ``<=`` convention by nudging each threshold
one float32 ulp down (``nextafter``), and mark the extracted model
``input_precision="float32"`` so runtimes quantize inputs to binary32
before comparing — reproducing XGBoost's routing exactly.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np

LEAF = -2


@dataclass
class ExtractedModel:
    """Float-precision tree ensemble extracted from a source model."""

    trees: list[dict]  # feature/threshold/left/right/value lists per tree
    base: float
    learning_rate: float
    family: str  # "sklearn" | "sklearn_hist" | "xgboost" | "lightgbm"
    input_precision: str = "float64"
    n_features: int = 0
    notes: list[str] = field(default_factory=list)


def score_float(extracted: ExtractedModel, x) -> float:
    """Reference float scorer used for extraction validation only."""
    s = extracted.base
    lr = extracted.learning_rate
    for tree in extracted.trees:
        feature = tree["feature"]
        threshold = tree["threshold"]
        left = tree["left"]
        right = tree["right"]
        node = 0
        while feature[node] != LEAF:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        s += lr * tree["value"][node]
    return float(s)


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------


def _is_sklearn_gbm(model) -> bool:
    return hasattr(model, "estimators_")


def _is_sklearn_hist(model) -> bool:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:  # pragma: no cover
        return False
    return isinstance(model, HistGradientBoostingRegressor)


def _is_xgboost(model) -> bool:
    try:
        import xgboost as xgb
    except ImportError:
        return False
    return isinstance(
        model, (xgb.XGBRegressor, xgb.XGBClassifier, xgb.XGBRanker, xgb.XGBRFRegressor, xgb.Booster)
    )


def _is_lightgbm(model) -> bool:
    try:
        import lightgbm as lgb
    except ImportError:
        return False
    return isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier, lgb.LGBMRanker, lgb.Booster))


# ---------------------------------------------------------------------------
# sklearn
# ---------------------------------------------------------------------------


def _extract_sklearn(model) -> ExtractedModel:
    if hasattr(model, "init_") and hasattr(model.init_, "constant_"):
        base = float(np.ravel(model.init_.constant_)[0])
    else:
        base = 0.0
    lr = float(getattr(model, "learning_rate", 1.0))

    trees = []
    for est in model.estimators_.ravel():
        t = est.tree_
        trees.append(
            {
                "feature": [int(v) if v >= 0 else LEAF for v in t.feature],
                "threshold": [float(v) for v in t.threshold],
                "left": [int(v) for v in t.children_left],
                "right": [int(v) for v in t.children_right],
                "value": [float(v) for v in t.value[:, 0, 0]],
            }
        )
    n_features = int(model.n_features_in_)
    return ExtractedModel(trees, base, lr, "sklearn", "float64", n_features)


# ---------------------------------------------------------------------------
# sklearn HistGradientBoosting
# ---------------------------------------------------------------------------


def _extract_sklearn_hist(model) -> ExtractedModel:
    """HistGradientBoostingRegressor — the constrained-whitebox backend.

    Walks the private ``_predictors`` structure (version-fragile by
    nature; the random-row parity gate in ``validate_extraction`` turns
    any sklearn-internals change into a loud failure instead of silent
    drift). Leaf values arrive pre-shrunk, so ``learning_rate`` is 1.0;
    the base is ``_baseline_prediction``. Thresholds are float64 and the
    split convention is ``x <= threshold -> left``, matching ours.
    """
    n_features = int(model.n_features_in_)
    is_cat = getattr(model, "is_categorical_", None)
    if is_cat is not None and any(is_cat):
        # Categorical splits are bitset-based AND remap feature_idx inside
        # the predictor nodes — both break the artifact's numeric-threshold
        # tree shape, so refuse at the model level before touching nodes.
        raise ValueError(
            "categorical splits in HistGradientBoosting are not supported; "
            "encode categoricals numerically before distilling"
        )
    trees = []
    missing_right_nodes = 0
    for predictors in model._predictors:
        if len(predictors) != 1:
            raise ValueError("multi-output HistGradientBoosting models are not supported")
        nodes = predictors[0].nodes
        n = len(nodes)
        feature = [LEAF] * n
        threshold = [0.0] * n
        left = [-1] * n
        right = [-1] * n
        value = [0.0] * n
        for i in range(n):
            node = nodes[i]
            if bool(node["is_leaf"]):
                value[i] = float(node["value"])
                continue
            if bool(node["is_categorical"]):
                raise ValueError(
                    "categorical splits in HistGradientBoosting are not supported; "
                    "encode categoricals numerically before distilling"
                )
            feature[i] = int(node["feature_idx"])
            threshold[i] = float(node["num_threshold"])
            left[i] = int(node["left"])
            right[i] = int(node["right"])
            if not bool(node["missing_go_to_left"]):
                missing_right_nodes += 1
        trees.append(
            {
                "feature": feature,
                "threshold": threshold,
                "left": left,
                "right": right,
                "value": value,
            }
        )
    notes: list[str] = []
    if missing_right_nodes:
        # Unlike XGBoost, HGB records a routing direction on every split even
        # when training saw no NaN, so this is a note, not a warning: the
        # artifact has no missing branch either way — missing_policy governs.
        notes.append(
            f"{missing_right_nodes} split(s) route missing values right in the source "
            "model; the artifact has no missing branch (missing_policy governs)."
        )
    base = float(np.ravel(model._baseline_prediction)[0])
    return ExtractedModel(trees, base, 1.0, "sklearn_hist", "float64", n_features, notes)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def _collect_xgb_nodes(node: dict, node_map: dict) -> None:
    node_map[int(node["nodeid"])] = node
    for child in node.get("children", []):
        _collect_xgb_nodes(child, node_map)


def _extract_xgboost(model) -> ExtractedModel:
    import xgboost as xgb

    booster = model.get_booster() if hasattr(model, "get_booster") else model

    feat_names = booster.feature_names
    if feat_names is not None:
        n_features = len(feat_names)
        name_to_idx = {name: i for i, name in enumerate(feat_names)}
    else:
        n_features = booster.num_features()
        name_to_idx = {f"f{i}": i for i in range(n_features)}

    trees = []
    notes: list[str] = []
    for tree_idx, tree_json in enumerate(booster.get_dump(dump_format="json")):
        root = json.loads(tree_json)
        node_map: dict = {}
        _collect_xgb_nodes(root, node_map)
        n = len(node_map)

        feature = [LEAF] * n
        threshold = [0.0] * n
        left = [-1] * n
        right = [-1] * n
        value = [0.0] * n

        for nid, node in node_map.items():
            if "leaf" in node:
                # Leaf values already include eta (XGBoost >= 1.7 applies Shrink).
                value[nid] = float(node["leaf"])
            else:
                feature[nid] = name_to_idx[str(node["split"])]
                # Strict `<` on float32 thresholds -> `<=` via one-ulp nudge in
                # float32 space, then widened to float64 for storage.
                raw_f32 = np.float32(float(node["split_condition"]))
                threshold[nid] = float(np.nextafter(raw_f32, np.float32(-np.inf)))
                left[nid] = int(node["yes"])
                right[nid] = int(node["no"])
                if int(node.get("missing", node["yes"])) != int(node["yes"]):
                    warnings.warn(
                        f"XGBoost tree {tree_idx}, node {nid}: missing values route to "
                        f"node {node.get('missing')}, not the yes/left child. The "
                        "compiled artifact has no missing-value branch; rely on the "
                        "artifact's missing_policy instead of NaN routing.",
                        stacklevel=3,
                    )

        trees.append(
            {
                "feature": feature,
                "threshold": threshold,
                "left": left,
                "right": right,
                "value": value,
            }
        )

    # Base score by residual against the booster's own margin (version-robust).
    draft = ExtractedModel(trees, 0.0, 1.0, "xgboost", "float32", n_features)
    x_dummy = np.zeros(n_features, dtype=np.float32)
    full_margin = float(booster.predict(xgb.DMatrix(x_dummy.reshape(1, -1)), output_margin=True)[0])
    base = full_margin - score_float(draft, [float(v) for v in x_dummy])

    return ExtractedModel(trees, base, 1.0, "xgboost", "float32", n_features, notes)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


def _flatten_lgbm_node(node: dict, records: list) -> int:
    idx = len(records)
    if "split_index" not in node:
        records.append({"is_leaf": True, "leaf_value": float(node.get("leaf_value", 0.0))})
        return idx
    record = {
        "is_leaf": False,
        "split_feature": int(node["split_feature"]),
        "threshold": float(node["threshold"]),
        "left_idx": None,
        "right_idx": None,
    }
    records.append(record)
    record["left_idx"] = _flatten_lgbm_node(node["left_child"], records)
    record["right_idx"] = _flatten_lgbm_node(node["right_child"], records)
    return idx


def _extract_lightgbm(model) -> ExtractedModel:
    booster = model.booster_ if hasattr(model, "booster_") else model
    dump = booster.dump_model()
    n_features = int(booster.num_feature())

    trees = []
    for tree_info in dump["tree_info"]:
        records: list = []
        _flatten_lgbm_node(tree_info["tree_structure"], records)
        n = len(records)
        feature = [LEAF] * n
        threshold = [0.0] * n
        left = [-1] * n
        right = [-1] * n
        value = [0.0] * n
        for i, rec in enumerate(records):
            if rec["is_leaf"]:
                value[i] = rec["leaf_value"]  # shrinkage already applied by LightGBM
            else:
                feature[i] = rec["split_feature"]
                threshold[i] = rec["threshold"]  # float64, `<=` convention matches
                left[i] = rec["left_idx"]
                right[i] = rec["right_idx"]
        trees.append(
            {
                "feature": feature,
                "threshold": threshold,
                "left": left,
                "right": right,
                "value": value,
            }
        )

    draft = ExtractedModel(trees, 0.0, 1.0, "lightgbm", "float64", n_features)
    x_dummy = [0.0] * n_features
    full_pred = float(booster.predict(np.zeros((1, n_features)), raw_score=True)[0])
    base = full_pred - score_float(draft, x_dummy)

    return ExtractedModel(trees, base, 1.0, "lightgbm", "float64", n_features)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_trees(model) -> ExtractedModel:
    """Extract a fitted model into float tree arrays; dispatches on family."""
    if _is_sklearn_hist(model):
        extracted = _extract_sklearn_hist(model)
        validate_extraction(extracted, model)
        return extracted
    if _is_sklearn_gbm(model):
        return _extract_sklearn(model)
    if _is_xgboost(model):
        extracted = _extract_xgboost(model)
        validate_extraction(extracted, model)
        return extracted
    if _is_lightgbm(model):
        extracted = _extract_lightgbm(model)
        validate_extraction(extracted, model)
        return extracted
    mod = type(model).__module__ or ""
    if any(fw in mod for fw in ("torch", "keras", "tensorflow")):
        raise TypeError(
            "Neural networks cannot be compiled directly to a tree artifact. "
            "Distill to a whitebox first: compileml.compile.train_whitebox()."
        )
    raise TypeError(
        f"Model type {type(model).__name__!r} is not compilable. "
        "Supported: sklearn gradient boosting, XGBoost, LightGBM."
    )


def validate_extraction(
    extracted: ExtractedModel, model, n_val: int = 200, atol: float = 1e-4
) -> None:
    """Score random rows through both paths; raise if they disagree."""
    rng = np.random.default_rng(0)
    if extracted.family == "xgboost":
        import xgboost as xgb

        booster = model.get_booster() if hasattr(model, "get_booster") else model
        X = rng.standard_normal((n_val, extracted.n_features)).astype(np.float32)
        ref = np.asarray(booster.predict(xgb.DMatrix(X), output_margin=True), dtype=float)
    elif extracted.family == "lightgbm":
        booster = model.booster_ if hasattr(model, "booster_") else model
        X = rng.standard_normal((n_val, extracted.n_features)).astype(np.float64)
        ref = np.asarray(booster.predict(X, raw_score=True), dtype=float)
    elif extracted.family == "sklearn_hist":
        X = rng.standard_normal((n_val, extracted.n_features)).astype(np.float64)
        ref = np.asarray(model.predict(X), dtype=float)
    else:
        return  # sklearn covered by unit tests against model.predict

    pred = np.array([score_float(extracted, [float(v) for v in X[i]]) for i in range(n_val)])
    max_diff = float(np.max(np.abs(ref - pred)))
    if max_diff > atol:
        raise ValueError(
            f"extraction parity failed for {extracted.family}: max_abs_diff={max_diff:.6g} "
            f"exceeds atol={atol:.6g} — please report this as a bug"
        )
