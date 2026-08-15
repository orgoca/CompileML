# Quickstart

Ten minutes from a fitted model to a verified, deployable decision artifact.

## Install

```bash
pip install compileml
```

## 1. Train a teacher, distill a whitebox

Any strong model can be the teacher — its only job is to produce a
probability-like latent. The whitebox is a small depth-2 gradient-boosted
regressor fitted to that latent; depth 2 is what makes attribution *exact*
(see [Exact attribution](concepts/attribution.md)).

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from compileml.compile import train_whitebox

teacher = GradientBoostingClassifier(n_estimators=300, max_depth=4).fit(X_train, y_train)

whitebox, fidelity = train_whitebox(
    X_train, teacher.predict_proba(X_train)[:, 1], n_estimators=120
)
print(fidelity)   # {'pearson': …, 'spearman': …, 'rmse': …}
latent = np.clip(whitebox.predict(X_train), 0.0, 1.0)
```

## 2. Build risk bands

```python
from compileml.bands import monotone_quantile_bands

bands = monotone_quantile_bands(latent, y_train, n_bands=10)
print(bands.metadata["empirical_bad_rate"])   # rises across bands
```

Other builders: `quantile_bands` (no outcomes needed), and the
search-and-certify pair `semantic_bands` / `governance_bands` that *discover*
how many bands the data statistically supports — see [Risk bands](concepts/bands.md).

## 3. Write the reason dictionary

This is your institutional language — CompileML computes impacts, you say what
they mean to a customer ([full how-to](howto/reason-codes.md)):

```python
reasons = {
    "UTILIZATION": {
        "code": "HIGH_UTILIZATION",
        "negative": "Credit utilization is high relative to available limits.",
        "positive": "Credit utilization is well managed.",
    },
    # … one entry per feature
}
```

## 4. Compile

```python
from compileml.artifact import build_artifact, save_artifact

artifact = build_artifact(
    whitebox,
    feature_names,
    baseline=np.median(X_train, axis=0),   # imputation + attribution reference
    band_edges=bands,
    calibration_latent=latent,
    calibration_y=y_train,
    reasons=reasons,
    X_sample=X_train[:500],                # enables the quantization report
)
save_artifact(artifact, "decision.json")
print(artifact["artifact_hash"])
```

Compile warns if reason coverage is below 100%, if the whitebox is deeper than
2, or if latents fall outside [0, 1].

## 5. Decide

```python
from compileml.runtime import load_artifact, decide

artifact = load_artifact("decision.json")   # SHA-256 verified
out = decide(artifact, X_test[0])
print(out["band"], out["pd"], out["latent_int"])
for r in out["reasons_negative"]:
    print(f"  {r['code']}: {r['message']} (impact {r['impact_int']})")
```

`decide(…, explain=False)` is the sub-millisecond path (score, band, PD only).

## 6. Validate, then deploy

```bash
compileml validate decision.json --csv holdout.csv --y-col DEFAULT --require-reasons
compileml export decision.json --target sql --out scorer.sql
compileml export decision.json --target cobol --out scorer.cob
```

The [validation framework](howto/validate.md) exits non-zero on any failed
check, so a deployment pipeline can gate on it. Both exports reproduce the
runtime's integers exactly — the SQL export is tested by executing it in a
real engine and asserting per-row integer equality.
