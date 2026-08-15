"""Build one artifact from fixed seeded data; print its hash.

CI runs this on Linux, macOS, and Windows with a pinned numerical stack and
requires the three hashes to be byte-identical — compile-side determinism,
end to end: data generation, model fit, distillation, banding, calibration,
quantization, canonical serialization.
"""

import numpy as np

from compileml.artifact import build_artifact
from compileml.bands import monotone_quantile_bands
from compileml.compile import train_whitebox

rng = np.random.default_rng(777)
n, p = 12_000, 10
X = rng.standard_normal((n, p))
teacher = 1 / (1 + np.exp(-(1.2 * X[:, 0] - 0.9 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3])))
y = (rng.random(n) < teacher).astype(int)

model, _ = train_whitebox(X, teacher, n_estimators=80, random_state=0)
latent = np.clip(model.predict(X), 0, 1)

artifact = build_artifact(
    model,
    [f"f{i}" for i in range(p)],
    np.median(X, axis=0),
    monotone_quantile_bands(latent, y, n_bands=10),
    calibration_latent=latent,
    calibration_y=y,
)
print(artifact["artifact_hash"])
