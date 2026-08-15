# Validate before deploying

The framework's defining property: **every check exercises the artifact
through the same runtime production uses.** There is no notebook shadow
implementation to drift out of sync with the deployed path — the historical
failure mode of validation tooling.

```python
from compileml.validate import validate_artifact

report = validate_artifact(
    "decision.json",              # or the dict; paths get load-and-verify
    X_val=X_holdout, y_val=y_holdout,
    model=whitebox,               # enables fidelity (check 3)
    latent_train=latent_train,    # enables churn baseline (check 6)
    require_full_reason_coverage=True,
)
assert report["all_pass"], report["checks"]
```

Or gate a pipeline on the CLI's exit code:

```bash
compileml validate decision.json --csv holdout.csv --y-col DEFAULT --require-reasons
```

## The eight checks

| # | Check | What it proves | Needs |
|---|---|---|---|
| 1 | integrity | hash verifies; structure valid; canonical JSON round-trip stable | nothing |
| 2 | reconciliation | the [spec §7.4](../ARTIFACT_SPEC.md) identity re-added on sample rows; residual exactly zero when exactness is claimed | X |
| 3 | fidelity | integer artifact within the quantization bound of the float model; rank order preserved (Spearman ≥ 0.999) | X + model |
| 4 | band properties | every band receives volume; latent resolution adequate | X |
| 5 | semantic monotonicity | bad rates non-decreasing across bands, measured **on the deployed integer path** | X + y |
| 6 | churn baseline | bootstrap ladder stability, measured with fixed-point edges | X + latent_train |
| 7 | explainability stability | top-k reason sets stable under small input perturbation, using the runtime's explainer | X |
| 8 | reason coverage | dictionary coverage of feature names; optional hard gate | nothing |

Checks lacking inputs **skip** (reported as skipped, not passed silently);
check 1 and check 8 always run.

## Evidence, not verdicts

Each check returns its numbers, not just a boolean — bad rates per band, worst
monotonicity drop, churn rate, mean Jaccard, worst residual — so a validation
report is reviewable, arguable material rather than a green light to trust.

## Sample-size guidance

Empirical checks need volume. Rule of thumb: a few hundred observations per
band before bad-rate monotonicity is meaningful; at 50 per band, sampling
noise alone can exceed the default 0.01 tolerance. The framework will fail
honestly on noise — give it enough data to fail only on signal.
