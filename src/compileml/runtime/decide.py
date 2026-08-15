"""The full decision pipeline: score → band → calibrate → explain.

``decide()`` is the reference implementation of a conforming CompileML
runtime. Every exporter target (COBOL, SQL, …) must reproduce its
integer outputs bit-for-bit.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from time import perf_counter

from compileml.runtime.bands import band_label
from compileml.runtime.calibrate import PD_SCALE, calibrate_ppm
from compileml.runtime.explain import contributions_half_micro, display_impacts, format_reasons
from compileml.runtime.io import ArtifactError
from compileml.runtime.score import latent_from_raw, score_micro


def _apply_precision(values: list[float], precision: str) -> list[float]:
    """Round values to IEEE binary32 when the artifact demands it (spec §4)."""
    if precision != "float32":
        return values
    return [struct.unpack("<f", struct.pack("<f", v))[0] for v in values]


def _prepare_row(artifact: dict, features: Sequence[float | None]) -> list[float]:
    """Apply the artifact's missing-value policy and input precision (spec §4, §8)."""
    names = artifact["features"]["names"]
    if len(features) != len(names):
        raise ArtifactError(f"expected {len(names)} features, got {len(features)}")

    baseline = artifact["features"]["baseline"]
    policy = artifact["features"].get("missing_policy", "baseline")
    row: list[float] = []
    for j, value in enumerate(features):
        missing = value is None or value != value  # None or NaN
        if not missing:
            row.append(float(value))
        elif policy == "baseline":
            row.append(float(baseline[j]))
        else:  # "reject"
            raise ArtifactError(
                f"missing value for feature {names[j]!r} and missing_policy is 'reject'"
            )
    return _apply_precision(row, artifact["model"].get("input_precision", "float64"))


def decide(
    artifact: dict,
    features: Sequence[float | None],
    *,
    top_k: int | None = None,
    explain: bool = True,
    include_contributions: bool = False,
) -> dict:
    """Run the complete decision for one row and return the payload dict.

    With ``explain=False`` this is the sub-millisecond score path: latent,
    band, and calibrated PD only. With ``explain=True`` it adds the exact
    integer attribution and reason blocks (O(n_features^2) traversals).
    """
    t0 = perf_counter()
    x = _prepare_row(artifact, features)

    model = artifact["model"]
    scale = int(artifact["scale"])
    micro_scale = int(model["micro_scale"])
    ratio = micro_scale // scale

    raw_micro = score_micro(model, x)
    latent_micro, latent_int = latent_from_raw(raw_micro, micro_scale, scale)
    band_idx, band = band_label(latent_int, artifact["bands"])
    pd_ppm = calibrate_ppm(latent_micro, artifact.get("calibration"), micro_scale)

    payload = {
        "band": band,
        "band_idx": band_idx,
        "pd_ppm": pd_ppm,
        "pd": pd_ppm / PD_SCALE,
        "latent_int": latent_int,
        "latent_micro": latent_micro,
        "raw_micro": raw_micro,
        "scale": scale,
        "micro_scale": micro_scale,
        "artifact_hash": artifact.get("artifact_hash"),
    }

    if explain:
        baseline = _apply_precision(
            [float(b) for b in artifact["features"]["baseline"]],
            model.get("input_precision", "float64"),
        )
        c2, full, fbase, residual2 = contributions_half_micro(model, x, baseline)
        impact_int = display_impacts(c2, full, fbase, residual2, ratio)

        runtime_cfg = artifact.get("runtime", {})
        k = int(top_k if top_k is not None else runtime_cfg.get("top_k", 5))
        reasons_negative, reasons_positive = format_reasons(
            c2,
            impact_int,
            artifact["features"]["names"],
            artifact.get("reasons") or {},
            artifact["features"].get("display_names") or {},
            k,
        )
        payload.update(
            {
                "attribution": "pairwise_interaction_int",
                "baseline_micro": fbase,
                "attribution_sum_half_micro": sum(c2),
                "attribution_residual_half_micro": residual2,
                "exact_attribution": residual2 == 0,
                "reasons_negative": reasons_negative,
                "reasons_positive": reasons_positive,
            }
        )
        if include_contributions:
            names = artifact["features"]["names"]
            payload["contributions"] = [
                {
                    "index": j,
                    "feature": str(names[j]),
                    "impact_half_micro": int(c2[j]),
                    "impact_int": int(impact_int[j]),
                }
                for j in range(len(c2))
            ]

    payload["elapsed_ms"] = (perf_counter() - t0) * 1000.0
    return payload
