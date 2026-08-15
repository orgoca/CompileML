"""Integer probability calibration (ARTIFACT_SPEC.md §6).

The calibration table is an isotonic step or piecewise-linear function
stored as integer thresholds (``f_micro``) and integer outputs
(``pd_ppm``, parts-per-million). Interpolation, when enabled, is exact
integer arithmetic — no float in the loop.
"""

from __future__ import annotations

from bisect import bisect_right

from compileml.runtime._intmath import div_rha

PD_SCALE = 1_000_000  # pd_ppm denominator (parts per million)


def calibrate_ppm(latent_micro: int, calibration: dict | None, micro_scale: int) -> int:
    """Calibrated probability of default in ppm for a clamped latent."""
    if not calibration:
        return div_rha(latent_micro * PD_SCALE, micro_scale)

    f = calibration["f_micro"]
    pd = calibration["pd_ppm"]
    k = len(f)
    if latent_micro <= f[0]:
        return int(pd[0])
    if latent_micro >= f[k - 1]:
        return int(pd[k - 1])

    hi = bisect_right(f, latent_micro)
    lo = hi - 1
    if calibration.get("mode", "linear_int") == "step":
        return int(pd[lo])

    num = (pd[hi] - pd[lo]) * (latent_micro - f[lo])
    den = f[hi] - f[lo]
    return int(pd[lo]) + div_rha(num, den)
