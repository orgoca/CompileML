"""Assemble, hash, and save CompileML decision artifacts (spec §3, §9)."""

from __future__ import annotations

import json
import warnings
from os import PathLike

import numpy as np

from compileml import __version__
from compileml.artifact.calibration import fit_isotonic_table
from compileml.compile.extract import extract_trees, score_float
from compileml.compile.monotone import normalize_constraints, verify_monotone_constraints
from compileml.compile.quantize import (
    max_depth,
    quantization_error_bound,
    quantize_model,
    rha,
)
from compileml.runtime.io import ARTIFACT_TYPE, SCHEMA_VERSION, canonical_hash, validate_structure
from compileml.runtime.score import score_micro


def _band_edges_int(band_edges, scale: int) -> list[int]:
    """Convert float band edges to strictly increasing fixed-point integers."""
    edges_int = [rha(float(e) * scale) for e in band_edges]
    collisions = [
        (band_edges[i], band_edges[i + 1])
        for i in range(len(edges_int) - 1)
        if edges_int[i + 1] <= edges_int[i]
    ]
    if collisions:
        raise ValueError(
            f"band edges collide after fixed-point conversion at scale={scale}: "
            f"{collisions}. Use fewer bands or a larger scale."
        )
    return edges_int


def build_artifact(
    model,
    feature_names,
    baseline,
    band_edges,
    *,
    band_labels=None,
    calibration_latent=None,
    calibration_y=None,
    calibration: dict | None = None,
    calibration_mode: str = "linear_int",
    reasons: dict | None = None,
    display_names: dict | None = None,
    feature_meta: list | None = None,
    missing_policy: str = "baseline",
    monotone_constraints=None,
    metadata: dict | None = None,
    scale: int = 1000,
    micro_scale: int = 1_000_000,
    top_k: int = 5,
    threshold_decimals: int | None = None,
    X_sample=None,
) -> dict:
    """Compile a fitted model into a complete, hashed decision artifact.

    Args:
        model: Fitted sklearn GBM, XGBoost, or LightGBM model whose raw
            output is a probability-like latent in [0, 1] (distilled
            whiteboxes always satisfy this; classifiers emitting log-odds
            margins must be distilled first via ``train_whitebox``).
        feature_names: Feature order the model was trained on.
        baseline: Reference row (typically imputer medians): imputation
            values under missing_policy="baseline" and the attribution
            reference point.
        band_edges: Float latent band edges (from compileml.bands builders),
            converted here to the fixed-point ladder.
        band_labels: Labels per band; defaults to G01..Gnn.
        calibration_latent: Latent sample used to fit the isotonic PD table
            (alternatively pass a prebuilt ``calibration`` block).
        calibration_y: Binary outcomes aligned with ``calibration_latent``.
        reasons: Reason dictionary mapping feature name -> {code, negative,
            positive, suppress}. **User-supplied content**: without an entry
            a feature falls back to generic messages that are not suitable
            for consumer-facing notices. Coverage below 100% warns and is
            recorded in metadata (spec §7.6).
        missing_policy: "baseline" (impute at decision time) or "reject".
        monotone_constraints: Declared directions per feature: a sequence of
            -1/0/+1 in feature order, or a dict keyed by feature name or
            index. The compiled integer trees are *verified* against the
            declaration (spec §3.1) — any violation raises, whatever
            trainer produced the model — and the signs are recorded in the
            artifact under ``model.monotone_constraints``, covered by the
            hash. Validation check 9 re-verifies on the artifact alone.
        X_sample: Optional sample rows; enables the measured quantization
            report and the latent-range check.

    Returns:
        The artifact dict, hashed and structurally validated.
    """
    if missing_policy not in ("baseline", "reject"):
        raise ValueError("missing_policy must be 'baseline' or 'reject'")
    if micro_scale % scale != 0:
        raise ValueError("micro_scale must be an integer multiple of scale")

    # Accept a BandSpec (from compileml.bands builders) in place of raw edges.
    bands_meta: dict = {}
    if hasattr(band_edges, "edges") and hasattr(band_edges, "labels"):
        spec = band_edges
        if band_labels is None:
            band_labels = list(spec.labels)
        bands_meta = dict(getattr(spec, "metadata", {}) or {})
        band_edges = list(spec.edges)

    names = [str(n) for n in feature_names]
    base_vals = [float(b) for b in np.asarray(baseline, dtype=float).reshape(-1)]
    if len(names) != len(base_vals):
        raise ValueError("feature_names and baseline must have the same length")

    # --- extract + quantize -------------------------------------------------
    extracted = extract_trees(model)
    if extracted.n_features and extracted.n_features != len(names):
        raise ValueError(f"model expects {extracted.n_features} features, got {len(names)} names")
    if threshold_decimals is not None:
        # Quantize split thresholds for decimal-arithmetic targets (spec §11).
        # Every runtime — Python included — then compares identical values;
        # the quantization report below measures the routing cost.
        for tree in extracted.trees:
            tree["threshold"] = [round(t, int(threshold_decimals)) for t in tree["threshold"]]
    model_int = quantize_model(extracted, micro_scale=micro_scale)

    # --- monotone constraints: verify against the shipped trees, then record --
    cst = normalize_constraints(monotone_constraints, len(names), feature_names=names)
    if cst is not None:
        report = verify_monotone_constraints(model_int, cst)
        if not report["ok"]:
            constrained = [names[i] for i, sign in enumerate(cst) if sign]
            raise ValueError(
                f"monotone constraint violated by the compiled trees: "
                f"{report['n_violations']} violation(s) across {constrained}. "
                "First examples: "
                f"{report['violations'][:3]}. Retrain with "
                "train_whitebox(..., monotone_constraints=...) to enforce them."
            )
        model_int["monotone_constraints"] = cst

    depth = max_depth(model_int)
    if depth > 2:
        warnings.warn(
            f"whitebox depth is {depth} (> 2): attribution will carry a nonzero "
            "residual and exact_attribution will be False (spec §7.3).",
            stacklevel=2,
        )

    # --- quantization + latent-range report ---------------------------------
    quant_report = {"error_bound": quantization_error_bound(model_int)}
    if threshold_decimals is not None:
        quant_report["threshold_decimals"] = int(threshold_decimals)
    if X_sample is not None:
        X_arr = np.asarray(X_sample, dtype=float)
        float_scores = np.array([score_float(extracted, [float(v) for v in row]) for row in X_arr])
        int_scores = (
            np.array([score_micro(model_int, [float(v) for v in row]) for row in X_arr])
            / micro_scale
        )
        quant_report["measured_max_error"] = float(np.max(np.abs(float_scores - int_scores)))
        outside = float(np.mean((float_scores < 0.0) | (float_scores > 1.0)))
        quant_report["share_outside_unit_interval"] = outside
        if outside > 0.01:
            warnings.warn(
                f"{outside:.1%} of sample latents fall outside [0, 1] before clamping. "
                "The artifact contract expects probability-like latents; distill "
                "margin-space models first (train_whitebox).",
                stacklevel=2,
            )

    # --- calibration ---------------------------------------------------------
    if calibration is None and calibration_latent is not None:
        if calibration_y is None:
            raise ValueError("calibration_y is required when calibration_latent is given")
        calibration = fit_isotonic_table(
            calibration_latent, calibration_y, micro_scale=micro_scale, mode=calibration_mode
        )

    # --- bands ----------------------------------------------------------------
    edges_int = _band_edges_int(band_edges, scale)
    n_bands = len(edges_int) - 1
    labels = (
        [str(label) for label in band_labels]
        if band_labels is not None
        else [f"G{i + 1:02d}" for i in range(n_bands)]
    )
    if len(labels) != n_bands:
        raise ValueError(f"expected {n_bands} band labels, got {len(labels)}")

    # --- reason coverage (spec §7.6: user-supplied content) -------------------
    reasons = dict(reasons or {})
    covered = [n for n in names if n in reasons]
    uncovered = [n for n in names if n not in reasons]
    coverage = len(covered) / len(names) if names else 1.0
    if uncovered:
        warnings.warn(
            f"reason dictionary covers {len(covered)}/{len(names)} features "
            f"({coverage:.0%}). Uncovered features fall back to generic messages "
            f"unsuitable for consumer-facing notices: {uncovered}",
            stacklevel=2,
        )

    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scale": int(scale),
        "model": model_int,
        "calibration": calibration,
        "bands": {
            "edges_int": edges_int,
            "labels": labels,
            "boundary": "left_closed_right_open",
        },
        "features": {
            "names": names,
            "baseline": base_vals,
            "missing_policy": missing_policy,
            "display_names": {str(k): str(v) for k, v in (display_names or {}).items()},
            "meta": _plain(feature_meta or []),
        },
        "reasons": _plain(reasons),
        "runtime": {
            "attribution": "pairwise_interaction_int",
            "top_k": int(top_k),
            "whitebox_max_depth": depth,
            "exact_attribution": depth <= 2,
        },
        "metadata": {
            **_plain(metadata or {}),
            "compileml_version": __version__,
            "model_family": extracted.family,
            "n_trees": len(model_int["trees"]),
            "reason_coverage": round(coverage, 4),
            "quantization": _plain(quant_report),
            **({"bands": _plain(bands_meta)} if bands_meta else {}),
        },
    }
    artifact["artifact_hash"] = canonical_hash(artifact)
    validate_structure(artifact)
    return artifact


def save_artifact(artifact: dict, path: str | PathLike, *, indent: int | None = 2) -> None:
    """Write an artifact to JSON (hash already embedded; loaders verify it)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=indent, ensure_ascii=False)


def _plain(value):
    """Recursively convert numpy scalars/arrays to JSON-native Python types."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
