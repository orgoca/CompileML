"""Zero-churn artifact recalibration — the retraining story (spec §6, §9).

When the portfolio drifts, PDs go stale but the model and the band edges
need not move. ``recalibrate_artifact`` refits the isotonic PD table and
refreshes per-band bad-rate metadata on fresh outcomes while keeping the
model and the fixed-point ladder byte-identical. The result is a new
hashed artifact that records its predecessor's hash — a provenance chain:

    artifact_v1 --(fresh outcomes)--> artifact_v2
    same model, same edges, same band assignments, new PDs, new hash

Because band assignment depends only on the model and edges, **no account
changes band** as a result of recalibration. That is the zero-churn
guarantee, and it is testable.
"""

from __future__ import annotations

import copy

import numpy as np

from compileml.artifact.calibration import fit_isotonic_table
from compileml.compile.quantize import rha
from compileml.runtime.bands import band_index
from compileml.runtime.io import canonical_hash


def recalibrate_artifact(
    artifact: dict,
    latent,
    y,
    *,
    mode: str | None = None,
    prior_strength: float = 0.0,
    prior_pi0: float | None = None,
) -> dict:
    """Refit calibration on fresh outcomes; model and band edges unchanged.

    Args:
        artifact: An existing decision artifact.
        latent: Fresh latent scores in [0, 1] (clipped), e.g. the artifact's
            own ``latent_micro / micro_scale`` on the new sample.
        y: Fresh binary outcomes aligned with ``latent``.
        mode: Calibration mode for the new table; defaults to the old one.
        prior_strength: Optional shrinkage weight pulling per-band empirical
            bad rates toward a prior for the band metadata refresh — small
            bands get stabilized rates.
        prior_pi0: The prior rate (defaults to the global bad rate).

    Returns:
        A new artifact dict with updated ``calibration``, refreshed band
        bad-rate metadata, ``metadata.recalibrated_from`` set to the old
        hash, and a new ``artifact_hash``.
    """
    F = np.clip(np.asarray(latent, dtype=float).reshape(-1), 0.0, 1.0)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if F.shape[0] != y_arr.shape[0]:
        raise ValueError("latent and y must have the same length")

    old_hash = artifact.get("artifact_hash")
    micro_scale = int(artifact["model"]["micro_scale"])
    scale = int(artifact["scale"])
    ratio = micro_scale // scale

    new = copy.deepcopy(artifact)

    # --- refit the decision-time PD table --------------------------------
    old_mode = (artifact.get("calibration") or {}).get("mode", "linear_int")
    new["calibration"] = fit_isotonic_table(
        F, y_arr, micro_scale=micro_scale, mode=mode or old_mode
    )

    # --- refresh per-band bad-rate metadata via the deployed integer path -
    edges_int = new["bands"]["edges_int"]
    n_bands = len(edges_int) - 1
    latent_int = [rha(float(v) * micro_scale) // ratio for v in F]
    idx = np.array([band_index(v, edges_int) for v in latent_int])
    counts = np.bincount(idx, minlength=n_bands)
    bad = np.bincount(idx, weights=y_arr, minlength=n_bands)
    global_pd = float(np.mean(y_arr))
    rate = np.where(counts > 0, bad / np.maximum(counts, 1), global_pd)
    if prior_strength > 0.0:
        pi0 = global_pd if prior_pi0 is None else float(prior_pi0)
        rate = (rate * counts + prior_strength * pi0) / (counts + prior_strength)

    new["metadata"] = dict(new.get("metadata") or {})
    new["metadata"]["recalibration"] = {
        "recalibrated_from": old_hash,
        "n_observations": int(F.shape[0]),
        "global_bad_rate": global_pd,
        "band_counts": [int(c) for c in counts],
        "band_bad_rate": [float(r) for r in rate],
        "prior_strength": float(prior_strength),
    }

    new["artifact_hash"] = canonical_hash(new)
    return new
