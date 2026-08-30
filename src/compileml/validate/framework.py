"""Pre-deployment validation — ten checks run against the artifact itself.

The defining property of this framework: **every check exercises the same
JSON document and the same runtime that production runs.** There is no
notebook-side shadow implementation to drift out of sync with the deployed
path.

Checks (each reports pass/skipped plus evidence):

  1. integrity                 hash verifies; structure valid; canonical
                               JSON round-trip is stable
  2. reconciliation            the spec §7.4 identity re-added on sample rows;
                               residual exactly zero when exactness is claimed
  3. fidelity                  integer artifact vs source float model within
                               the quantization bound; rank order preserved
  4. band_properties           every band actually receives volume; latent
                               resolution is adequate
  5. semantic_monotonicity     bad rates by band non-decreasing (tolerance),
                               measured on the deployed integer path
  6. churn_baseline            bootstrap ladder stability, measured with
                               fixed-point edges — the deployed semantics
  7. explainability_stability  top-k reason sets stable under small input
                               perturbation, using the runtime's explainer
  8. reason_coverage           reason dictionary coverage of feature names
  9. monotone_constraints      declared directions re-verified against the
                               shipped integer trees (skipped when the
                               artifact declares none)
 10. reference_floor           the artifact out-scores a reference model on
                               the same data — the floor that teacher
                               retention alone can never supply
"""

from __future__ import annotations

import json

import numpy as np

from compileml.compile.quantize import rha
from compileml.runtime.bands import band_index
from compileml.runtime.decide import decide
from compileml.runtime.io import canonical_hash, load_artifact, validate_structure, verify_artifact


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _latents_int(artifact: dict, X: np.ndarray) -> np.ndarray:
    """Display-scale latents for many rows via the deployed decision path."""
    return np.array(
        [decide(artifact, [float(v) for v in row], explain=False)["latent_int"] for row in X],
        dtype=int,
    )


def validate_artifact(
    artifact_or_path,
    X_val=None,
    y_val=None,
    *,
    model=None,
    latent_train=None,
    n_reconciliation: int = 30,
    n_stability: int = 50,
    monotonicity_tolerance: float = 0.01,
    churn_threshold: float = 0.15,
    jaccard_threshold: float = 0.5,
    noise_scale: float = 0.01,
    min_fidelity_spearman: float = 0.999,
    max_within_band_auc: float | None = None,
    require_full_reason_coverage: bool = False,
    reference=None,
    require_reference_floor: bool = False,
    seed: int = 42,
) -> dict:
    """Run the ten-check framework. Checks lacking inputs skip, not fail.

    Args:
        artifact_or_path: Artifact dict or path to its JSON (paths get the
            full load-and-verify treatment).
        X_val: Validation rows for the empirical checks (2–7).
        y_val: Binary outcomes aligned with ``X_val`` (checks 5).
        model: The source float model; enables the fidelity check (3).
        latent_train: Training latents; enables the churn baseline (6).
        require_full_reason_coverage: Make check 8 fail below 100% coverage
            (recommended for consumer-facing deployments).
        reference: A fitted ``compileml.reference.ReferenceModel`` or a bare
            Gini float, enabling check 10. Retention against a teacher is
            one-sided and cannot reveal that a plain logistic regression
            outscores the artifact; this is the other side of it. A champion
            scorecard's Gini is a better floor than a fitted one — pass the
            number.
        require_reference_floor: Make check 10 *fail* when the artifact does
            not clear the reference, rather than recording it as evidence
            (the ``max_within_band_auc`` precedent).

    Returns:
        {"all_pass": bool, "checks": {name: {"pass", "skipped", ...evidence}}}
    """
    rng = np.random.default_rng(seed)
    checks: dict[str, dict] = {}

    # ---------------------------------------------------------- 1 integrity
    if isinstance(artifact_or_path, (str, bytes)) or hasattr(artifact_or_path, "__fspath__"):
        artifact = load_artifact(artifact_or_path)  # raises on tamper
        hash_ok = True
    else:
        artifact = artifact_or_path
        validate_structure(artifact)
        hash_ok = verify_artifact(artifact)
    roundtrip = json.loads(json.dumps(artifact))
    roundtrip_ok = canonical_hash(roundtrip) == canonical_hash(artifact)
    checks["1_integrity"] = {
        "pass": bool(hash_ok and roundtrip_ok),
        "skipped": False,
        "hash_verified": bool(hash_ok),
        "json_roundtrip_stable": bool(roundtrip_ok),
        "artifact_hash": artifact.get("artifact_hash"),
    }

    scale = int(artifact["scale"])
    micro_scale = int(artifact["model"]["micro_scale"])
    n_bands = len(artifact["bands"]["labels"])
    X = None if X_val is None else np.asarray(X_val, dtype=float)
    y = None if y_val is None else np.asarray(y_val, dtype=float).reshape(-1)

    # ----------------------------------------------------- 2 reconciliation
    if X is not None:
        take = min(n_reconciliation, len(X))
        rows = X[rng.choice(len(X), size=take, replace=False)]
        claims_exact = bool(artifact["runtime"].get("exact_attribution"))
        identity_ok, exact_ok, worst_residual = True, True, 0
        for row in rows:
            out = decide(artifact, [float(v) for v in row], include_contributions=True)
            lhs = 2 * (out["raw_micro"] - out["baseline_micro"])
            rhs = out["attribution_sum_half_micro"] + out["attribution_residual_half_micro"]
            identity_ok &= lhs == rhs
            residual = abs(out["attribution_residual_half_micro"])
            worst_residual = max(worst_residual, residual)
            if claims_exact and residual != 0:
                exact_ok = False
        checks["2_reconciliation"] = {
            "pass": bool(identity_ok and exact_ok),
            "skipped": False,
            "rows_checked": int(take),
            "identity_holds": bool(identity_ok),
            "claims_exact_attribution": claims_exact,
            "worst_abs_residual_half_micro": int(worst_residual),
        }
    else:
        checks["2_reconciliation"] = {"pass": True, "skipped": True}

    # ----------------------------------------------------------- 3 fidelity
    if X is not None and model is not None:
        from scipy.stats import spearmanr

        from compileml.compile.extract import extract_trees, score_float
        from compileml.compile.quantize import quantization_error_bound

        extracted = extract_trees(model)
        # Mirror any threshold quantization the artifact was built with, so the
        # float reference routes rows the same way (spec §11).
        decimals = artifact.get("metadata", {}).get("quantization", {}).get("threshold_decimals")
        if decimals is not None:
            for tree in extracted.trees:
                tree["threshold"] = [round(t, int(decimals)) for t in tree["threshold"]]
        bound = quantization_error_bound(artifact["model"])
        float_scores = np.array([score_float(extracted, [float(v) for v in row]) for row in X])
        raw_micro = np.array(
            [decide(artifact, [float(v) for v in row], explain=False)["raw_micro"] for row in X]
        )
        max_err = float(np.max(np.abs(float_scores - raw_micro / micro_scale)))
        rho = float(spearmanr(float_scores, raw_micro)[0])
        checks["3_fidelity"] = {
            "pass": bool(max_err <= bound + 1e-12 and rho >= min_fidelity_spearman),
            "skipped": False,
            "max_abs_error": max_err,
            "error_bound": bound,
            "spearman_vs_float_model": rho,
            "spearman_threshold": min_fidelity_spearman,
        }
    else:
        checks["3_fidelity"] = {"pass": True, "skipped": True}

    # ---------------------------------------------------- 4 band properties
    latents = None
    if X is not None:
        latents = _latents_int(artifact, X)
        bands_seen = np.array([band_index(v, artifact["bands"]["edges_int"]) for v in latents])
        coverage_ok = len(np.unique(bands_seen)) == n_bands
        resolution = len(np.unique(latents)) / max(1, len(latents))
        checks["4_band_properties"] = {
            "pass": bool(coverage_ok and resolution > 0.01),
            "skipped": False,
            "bands_expected": n_bands,
            "bands_seen": int(len(np.unique(bands_seen))),
            "unique_latent_ratio": float(resolution),
        }
        if y is not None:
            # Band efficiency: how much discrimination the ladder discards,
            # and whether any band could still be split ("money on the
            # table"). Advisory evidence unless max_within_band_auc gates it.
            from compileml.bands.efficiency import band_efficiency

            latent_float = latents.astype(float) / scale
            eff = band_efficiency(latent_float, y.astype(int), artifact, n_boot=100, seed=seed)
            check4 = checks["4_band_properties"]
            check4["banding_gini_gap"] = eff["gini_gap"]
            check4["band_efficiency"] = eff["per_band"]
            worst = [p for p in eff["per_band"] if p["auc_ci"] is not None]
            worst_auc = max((p["auc_within"] for p in worst), default=None)
            check4["worst_within_band_auc"] = worst_auc
            check4["worst_band"] = eff["worst_band"]
            if max_within_band_auc is not None and worst_auc is not None:
                check4["max_within_band_auc"] = float(max_within_band_auc)
                check4["pass"] = bool(check4["pass"] and worst_auc <= max_within_band_auc)
    else:
        checks["4_band_properties"] = {"pass": True, "skipped": True}

    # ---------------------------------------------- 5 semantic monotonicity
    if X is not None and y is not None:
        bands_seen = np.array([band_index(v, artifact["bands"]["edges_int"]) for v in latents])
        rates = [
            float(np.mean(y[bands_seen == b])) if np.any(bands_seen == b) else float("nan")
            for b in range(n_bands)
        ]
        valid = np.array([r for r in rates if not np.isnan(r)])
        worst_drop = float(np.min(np.diff(valid))) if len(valid) >= 2 else 0.0
        checks["5_semantic_monotonicity"] = {
            "pass": bool(worst_drop >= -monotonicity_tolerance),
            "skipped": False,
            "bad_rate_by_band": {
                str(artifact["bands"]["labels"][b]): rates[b] for b in range(n_bands)
            },
            "worst_drop": worst_drop,
            "tolerance": monotonicity_tolerance,
        }
    else:
        checks["5_semantic_monotonicity"] = {"pass": True, "skipped": True}

    # -------------------------------------------------- 6 churn baseline
    if latent_train is not None and latents is not None:
        ltrain = np.asarray(latent_train, dtype=float).reshape(-1)

        def bootstrap_cutoffs():
            sample = rng.choice(ltrain, size=len(ltrain), replace=True)
            edges = np.quantile(sample, np.linspace(0, 1, n_bands + 1))
            return np.array([rha(float(e) * scale) for e in edges[1:-1]], dtype=int)

        cuts_a, cuts_b = bootstrap_cutoffs(), bootstrap_cutoffs()
        band_a = np.searchsorted(cuts_a, latents, side="right")
        band_b = np.searchsorted(cuts_b, latents, side="right")
        churn = float(np.mean(band_a != band_b))
        checks["6_churn_baseline"] = {
            "pass": bool(churn <= churn_threshold),
            "skipped": False,
            "churn_rate": churn,
            "churn_threshold": churn_threshold,
        }
    else:
        checks["6_churn_baseline"] = {"pass": True, "skipped": True}

    # ---------------------------------------- 7 explainability stability
    if X is not None:
        take = min(n_stability, len(X))
        pick = rng.choice(len(X), size=take, replace=False)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0
        jaccards = []
        for i in pick:
            row = X[i]
            base_out = decide(artifact, [float(v) for v in row])
            base_feats = [
                r["feature"] for r in base_out["reasons_negative"] + base_out["reasons_positive"]
            ]
            noisy = row + rng.normal(0.0, noise_scale * stds, size=row.shape)
            pert_out = decide(artifact, [float(v) for v in noisy])
            pert_feats = [
                r["feature"] for r in pert_out["reasons_negative"] + pert_out["reasons_positive"]
            ]
            jaccards.append(_jaccard(base_feats, pert_feats))
        mean_jaccard = float(np.mean(jaccards)) if jaccards else float("nan")
        checks["7_explainability_stability"] = {
            "pass": bool(np.isnan(mean_jaccard) or mean_jaccard >= jaccard_threshold),
            "skipped": False,
            "rows_checked": int(take),
            "mean_jaccard": mean_jaccard,
            "jaccard_threshold": jaccard_threshold,
            "noise_scale": noise_scale,
        }
    else:
        checks["7_explainability_stability"] = {"pass": True, "skipped": True}

    # ------------------------------------------------- 8 reason coverage
    names = artifact["features"]["names"]
    reasons = artifact.get("reasons") or {}
    uncovered = [n for n in names if n not in reasons]
    coverage = 1.0 - len(uncovered) / max(1, len(names))
    checks["8_reason_coverage"] = {
        "pass": bool(coverage == 1.0 or not require_full_reason_coverage),
        "skipped": False,
        "coverage": round(coverage, 4),
        "uncovered_features": uncovered,
        "required_full": require_full_reason_coverage,
    }

    # -------------------------------------------- 9 monotone constraints
    cst = artifact["model"].get("monotone_constraints")
    if cst:
        from compileml.compile.monotone import verify_monotone_constraints

        report = verify_monotone_constraints(artifact["model"], cst)
        checks["9_monotone_constraints"] = {
            "pass": bool(report["ok"]),
            "skipped": False,
            "constrained_features": [
                names[i] for i, sign in enumerate(cst) if sign and i < len(names)
            ],
            "n_violations": int(report["n_violations"]),
            "violations": report["violations"][:5],
        }
    else:
        checks["9_monotone_constraints"] = {"pass": True, "skipped": True}

    # ------------------------------------------------ 10 reference floor
    if reference is not None and X is not None and y is not None:
        from sklearn.metrics import roc_auc_score

        from compileml.reference.woe import reference_gini as _reference_gini

        if latents is None:
            latents = _latents_int(artifact, X)
        artifact_gini = 2 * float(roc_auc_score(y, latents.astype(float))) - 1
        ref_gini = _reference_gini(reference, X, y)
        clears = artifact_gini >= ref_gini
        checks["10_reference_floor"] = {
            "pass": bool(clears or not require_reference_floor),
            "skipped": False,
            "artifact_gini": round(artifact_gini, 4),
            "reference_gini": round(ref_gini, 4),
            "gini_vs_reference_pct": (
                round(100 * artifact_gini / ref_gini, 2) if ref_gini > 0 else None
            ),
            "clears_floor": bool(clears),
            "required": bool(require_reference_floor),
        }
    else:
        checks["10_reference_floor"] = {"pass": True, "skipped": True}

    return {"all_pass": all(c["pass"] for c in checks.values()), "checks": checks}
