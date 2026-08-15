"""Leaf quantization: float trees → the integer artifact model (spec §2.3).

This is the step that makes determinism literal. Each leaf becomes
``rha(value * learning_rate * micro_scale)`` — an integer — and the float
model is discarded. From here on the integer model *is* the model:
retention, parity, and every downstream number are properties of it.

The worst-case quantization error versus the float model is
``(n_trees + 1) / (2 * micro_scale)`` in latent units — at the default
micro_scale of 1e6, about 5e-5 for a 100-tree ensemble.
"""

from __future__ import annotations

import math

from compileml.compile.extract import LEAF, ExtractedModel


def rha(x: float) -> int:
    """Round half away from zero (spec §2.1)."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def quantize_model(extracted: ExtractedModel, *, micro_scale: int = 1_000_000) -> dict:
    """Quantize an extracted float model into the integer artifact model."""
    trees_int = []
    for tree in extracted.trees:
        feature = list(tree["feature"])
        value_micro = [
            (
                rha(tree["value"][i] * extracted.learning_rate * micro_scale)
                if feature[i] == LEAF
                else 0
            )
            for i in range(len(feature))
        ]
        trees_int.append(
            {
                "feature": feature,
                "threshold": [float(t) for t in tree["threshold"]],
                "left": list(tree["left"]),
                "right": list(tree["right"]),
                "value_micro": value_micro,
            }
        )

    return {
        "kind": "tree_ensemble_int",
        "micro_scale": int(micro_scale),
        "base_micro": rha(extracted.base * micro_scale),
        "n_features": int(extracted.n_features),
        "input_precision": extracted.input_precision,
        "trees": trees_int,
    }


def max_depth(model_int: dict) -> int:
    """Actual maximum tree depth, measured from the node arrays."""
    deepest = 0
    for tree in model_int["trees"]:
        feature = tree["feature"]
        left, right = tree["left"], tree["right"]
        stack = [(0, 0)]
        while stack:
            node, depth = stack.pop()
            if feature[node] == LEAF:
                deepest = max(deepest, depth)
            else:
                stack.append((left[node], depth + 1))
                stack.append((right[node], depth + 1))
    return deepest


def quantization_error_bound(model_int: dict) -> float:
    """Worst-case |float_latent − int_latent/micro_scale| (latent units)."""
    n_trees = len(model_int["trees"])
    return (n_trees + 1) / (2.0 * model_int["micro_scale"])
