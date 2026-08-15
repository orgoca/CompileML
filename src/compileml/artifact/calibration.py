"""Fit an isotonic calibration and freeze it as an integer table (spec §6)."""

from __future__ import annotations

import numpy as np

from compileml.compile.quantize import rha
from compileml.runtime.calibrate import PD_SCALE


def fit_isotonic_table(
    latent,
    y,
    *,
    micro_scale: int = 1_000_000,
    mode: str = "linear_int",
) -> dict:
    """Isotonic PD calibration frozen into integer thresholds.

    ``latent`` is the (clipped) model latent on a calibration sample; ``y``
    the binary outcomes. Returns the artifact's ``calibration`` block:
    strictly increasing ``f_micro``, non-decreasing ``pd_ppm``.
    """
    from sklearn.isotonic import IsotonicRegression

    if mode not in ("linear_int", "step"):
        raise ValueError("mode must be 'linear_int' or 'step'")

    x = np.clip(np.asarray(latent, dtype=float).reshape(-1), 0.0, 1.0)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if x.shape[0] != yy.shape[0]:
        raise ValueError("latent and y must have the same length")
    if x.shape[0] == 0:
        raise ValueError("latent must not be empty")

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, yy)

    f_micro: list[int] = []
    pd_ppm: list[int] = []
    for xf, yf in zip(iso.X_thresholds_, iso.y_thresholds_):
        f = rha(float(xf) * micro_scale)
        p = min(max(rha(float(yf) * PD_SCALE), 0), PD_SCALE)
        if f_micro and f == f_micro[-1]:
            # Thresholds that collide after rounding: keep the larger PD
            # (isotonic y is non-decreasing, so this is the later one).
            pd_ppm[-1] = max(pd_ppm[-1], p)
        else:
            f_micro.append(f)
            pd_ppm.append(p)

    return {"mode": mode, "f_micro": f_micro, "pd_ppm": pd_ppm}
