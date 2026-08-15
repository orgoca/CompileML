# Zero-churn recalibration

The retraining question every model committee asks: *what happens to existing
accounts when you update?* CompileML's answer: for calibration updates,
**nothing moves except the probabilities** — and that's provable, not policy.

## The mechanism

```python
from compileml.artifact import recalibrate_artifact

new_artifact = recalibrate_artifact(old_artifact, fresh_latents, fresh_outcomes)
```

`recalibrate_artifact` refits the isotonic PD table and refreshes per-band
bad-rate metadata on fresh outcomes while the **model and the band ladder stay
byte-identical**. Band assignment depends only on the model and the edges, so
no account can change band — the zero-churn guarantee is structural, and the
test suite proves it row by row.

## The provenance chain

The new artifact records its predecessor:

```json
"metadata": {
  "recalibration": {
    "recalibrated_from": "84372c36…",     // the old artifact's hash
    "n_observations": 96000,
    "global_bad_rate": 0.291,
    "band_counts": [ … ],
    "band_bad_rate": [ … ]
  }
}
```

Since hashes are the unit of governance, recalibration produces a chain:

```
artifact_v1  ──fresh outcomes──►  artifact_v2  ──fresh outcomes──►  artifact_v3
   hash A                            hash B                            hash C
                                 (records A)                       (records B)
```

Each link answers "same decisions, updated probabilities, here's the evidence"
— no timestamps to trust, just hashes to verify.

## Optional shrinkage

Small bands produce noisy empirical rates. `prior_strength` shrinks per-band
bad rates toward a prior (the global rate by default):

```python
recalibrate_artifact(artifact, latents, outcomes, prior_strength=50)
```

The decision-time PD (from the isotonic table) is unaffected by the shrinkage
option; it stabilizes the *reporting* metadata for small bands.

## When you actually need a new model

Recalibration handles level drift (PDs stale, rank order fine). If rank order
itself degrades — check 3 of the [validation framework](validate.md) against
fresh outcomes will show it — that's a retrain, which produces a genuinely new
artifact with new edges and a fresh governance cycle. CompileML makes the two
cases mechanically distinct: one preserves the model and ladder bytes, the
other doesn't, and the hashes say which happened.
