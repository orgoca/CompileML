# CompileML

[![ci](https://github.com/orgoca/CompileML/actions/workflows/ci.yml/badge.svg)](https://github.com/orgoca/CompileML/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/compileml)](https://pypi.org/project/compileml/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/compileml/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Keep the predictive power of a tree ensemble. Deploy it as transparent decision logic with no model runtime.**

## Why this exists

Risk teams are usually offered a bad choice.

A traditional scorecard is transparent, reproducible, and deploys anywhere, but leaves predictive power on the table. A tree ensemble predicts better, then arrives with a Python environment, a serving stack, post-hoc explanations, and a model no validator can reproduce independently.

CompileML removes that choice.

Train the strongest teacher you can — XGBoost, LightGBM, NN. CompileML projects its predictive structure into a shallow, integer-valued whitebox and packages the complete decision into one hashed artifact: score, calibrated probability, risk bands, reason codes, attribution.

Then the teacher is discarded. It was a means, not a deliverable, and it never reaches production.

What ships instead is decision logic built from integer addition, comparison, and table lookup. It runs under CompileML's standard-library-only Python implementation, an ordinary application, a small Lambda, a batch job or you export it as standalone SQL or COBOL and run it directly where the decision already happens. No XGBoost in production, no scikit-learn, no scoring service to operate. In exported form there is no model runtime at all.

This is not a lighter way to serve the black box. The black box was used as a teacher and then compiled out of the system.

That buys three things, and I would not trade any one of them for the other two: the artifact reproduces to the integer on any machine, it explains itself by arithmetic rather than approximation, and it lands where the decision already runs. The sections below are how each one is enforced.

The benchmark puts the cost at about 2% of the teacher's Gini. In exchange, at whitebox depth two or less, every decision reconstructs exactly from a printed scorecard table, and every explanation adds back to the production score with nothing left over.

## Quick example

Train however you want. The example below uses a strong model as a teacher and distills it into a shallow whitebox:

```python
from compileml.compile import train_whitebox
from compileml.bands import monotone_quantile_bands
from compileml.artifact import build_artifact, save_artifact

whitebox, fidelity = train_whitebox(
    X_train,
    teacher.predict_proba(X_train)[:, 1],
)

latent = whitebox.predict(X_train).clip(0, 1)

bands = monotone_quantile_bands(
    latent,
    y_train,
    n_bands=10,
)

artifact = build_artifact(
    whitebox,
    feature_names,
    baseline=medians,
    band_edges=bands,
    calibration_latent=latent,
    calibration_y=y_train,
    reasons=REASON_DICTIONARY,
)

save_artifact(artifact, "decision.json")
```

Production does not need the training stack:

```python
from compileml.runtime import load_artifact, decide

artifact = load_artifact("decision.json")
decision = decide(artifact, applicant_row)

# {
#   "band": "G07",
#   "pd": 0.1284,
#   "latent_int": 146,
#   "reasons_negative": [
#       {
#           "code": "HIGH_UTILIZATION",
#           "message": "…",
#           "impact_int": 56,
#       }
#   ],
#   "reasons_positive": [...],
#   "artifact_hash": "84372c36…",
# }
```

The runtime imports nothing outside the Python standard library.

Or skip the Python runtime entirely:

```bash
compileml export decision.json --target sql   --out scorer.sql
compileml export decision.json --target cobol --out scorer.cob
```

## The three objections, answered

I have watched good models become much less impressive on the way to production.

The model starts in Python. Someone rewrites it in SQL. Someone else builds the bands in a spreadsheet. Calibration lives in another script. Reason codes are produced through a separate explanation process. Six months later, everybody is discussing "the model," but they are no longer talking about exactly the same thing.

Each of the three standing objections has a structural answer.

### Stability: scores drift

Floating-point arithmetic is not a reassuring foundation for a decision that must be reproduced across languages and systems. Small differences in accumulation, precision, or implementation can move a score near a boundary.

A credit score should be a fact, not a distribution over environments.

CompileML quantizes model leaves once, at compile time. After that, scoring is integer addition, banding is integer comparison, and calibration is integer table lookup. Recalibration refits probabilities while the model and band edges stay byte-identical, so updating a PD table cannot move a single account between bands.

### Explainability: explanations do not reconcile

Post-hoc explainers are useful, but an explanation of a regulated decision should not merely resemble the decision.

CompileML computes attribution from the compiled model in integer units. The feature impacts, baseline, and residual satisfy a reconciliation identity that validation can add back independently.

For whiteboxes of depth two or less, the pairwise decomposition is complete and the residual is exactly zero. The runtime refuses to emit an explanation that fails to reconcile.

### Deployability: the deployment stack is not the modeling stack

Banks and other large institutions run important decisions on SQL systems, core platforms, and mainframes. Requiring the entire training environment in production is often unrealistic and sometimes unnecessary.

CompileML moves complexity to compile time and leaves production with a small, explicit artifact.

## It compiles to a scorecard

At whitebox depth ≤ 2, the compiled artifact collapses into a classic points scorecard — exactly, not as an approximation.

```bash
compileml scorecard decision.json --format csv --out scorecard.csv
```

Depth 1 produces the familiar form: per feature, a bin and its points. Depth 2 adds explicit pairwise interaction grids over the union of the relevant thresholds. Points are the artifact's own integers, and the identity

```text
base_points + Σ main_effect(x) + Σ interaction(x) == score
```

holds bit-for-bit on every row. `score_from_scorecard()` re-derives any production decision from the printed tables alone, and the test suite asserts it.

Hand the CSV to a validator and they can reproduce production scores in a spreadsheet. Above depth 2 no clean scorecard exists, and the tool raises instead of approximating — the same boundary as exact attribution, for the same reason.

## What the artifact guarantees

Given the same artifact and the same input values, CompileML produces the same governed integer outputs across supported runtimes.

The repository tests this rather than asking you to take it on faith:

* SQL output is executed in SQLite and compared row by row with the Python runtime.
* Generated COBOL is compiled and run in CI, then checked against the reference implementation.
* The same seeded artifact is built on Linux, macOS, and Windows and the hashes are compared.
* Committed reference decisions are replayed on every OS and Python version in the matrix.
* Attribution is added back to the decision during validation.
* Recalibration tests verify that the model and band edges remain unchanged.
* Scorecard tables are re-summed against the runtime's own integers.
* The standard-library-only runtime is enforced by inspecting its imports.

The artifact includes a SHA-256 hash. Loaders verify it by default and reject a document whose contents no longer match the stored hash. This detects modification or corruption; it is an integrity check, not a cryptographic signature of who produced the artifact.

## Measured performance

The committed benchmark uses deterministic synthetic credit data with 40,000 rows and 23 features. It runs on a consumer laptop through the pure-Python runtime.

You can reproduce every number with:

```bash
python benchmarks/run_benchmarks.py
```

| Metric                                     |                      Value |
| ------------------------------------------ | -------------------------: |
| Teacher Gini, 300-tree GBM                 |                      0.667 |
| **Compiled integer artifact Gini**         | **0.653 — 97.9% retained** |
| Band-ordinal Gini, 10 bands                |     0.647 — 97.0% retained |
| Spearman correlation, teacher vs. artifact |                      0.977 |
| Score + band + calibrated PD               |         **0.03 ms median** |
| Score + band + calibrated PD, p95          |                    0.04 ms |
| Full exact explanation, 23 features        |              8.4 ms median |
| Band assignment alone                      |                     0.2 µs |
| Artifact size                              |                      97 KB |
| Identical hash on rebuild                  |                        Yes |

That 2% of Gini is the price of everything above it. It is stated rather than hidden, and it is reproducible on your own data with `compileml.tune.sweep_whitebox`.

One honest qualification: scoring is very fast; full explanation is not equally cheap.

The exact pairwise decomposition requires:

```text
1 + p + p(p−1)/2
```

ensemble traversals for `p` features. It is exact rather than sampled, and that has a cost.

In practice: explain everything. A few milliseconds per decision is real-time for credit decisioning — the bureau pull costs more — and complete attribution on every decision is what turns portfolio questions (marginal analysis, driver drift, fairness cuts) into census facts instead of sample estimates. It also means every production decision carries its own explanation in the record, computed at decision time under the same artifact hash.

The quadratic cost matters in one place: re-explaining an entire book in batch, or artifacts with very wide feature sets. That is what the leaf-time roadmap item addresses — not live latency, which was never the constraint.

## Choosing the configuration

The two capacity knobs are not symmetric, and this is the single most useful thing to know before tuning.

`n_estimators` buys fidelity at a linear cost in artifact size and explanation time, and costs nothing else. Determinism, portability, and exact attribution are unaffected at any tree count.

`max_depth` buys fidelity per tree, but above 2 it takes the exactness guarantee with it: attribution residuals appear and no clean scorecard exists.

Spend on trees. Be stingy with depth.

Both are measurable rather than guessable:

```python
from compileml.tune import sweep_whitebox, sweep_bands

sweep_whitebox(X, teacher_latent, y, X_val=X_val, y_val=y_val, teacher_latent_val=t_val)
sweep_bands(latent, y, k_grid=(4, 6, 8, 10, 12, 16))
```

For banding, `band_efficiency()` reports what the ladder discards — the Gini gap against the continuous score — and, per band, whether the score still ranks risk internally. A band that can still separate outcomes is a refinement opportunity; a band that cannot is a band you have used up. The [tuning guide](docs/howto/tuning.md) walks through both.

## What CompileML is not

CompileML is not a new training framework. Use XGBoost, LightGBM, scikit-learn, or another teacher that can be distilled into the supported whitebox representation.

It is not a promise that your data pipelines are identical. Determinism means:

```text
same input values + same artifact = same governed outputs
```

Producing the same input values across systems remains the caller's responsibility.

It is also not a compliance certification. No library can certify an institution's model, data, policy language, or governance process.

CompileML is infrastructure intended to make those things inspectable instead of asking validators to trust a chain of separate implementations.

## Reason codes belong to the institution

CompileML can determine which features moved a decision and by how much. It cannot decide how your institution should explain that result to a customer.

That language is policy, not mathematics.

You provide the reason dictionary:

```python
REASON_DICTIONARY = {
    "BILLS_PAID_LATE": {
        "code": "LATE_PAYMENTS",
        "negative": "Recent payments were made after their due date.",
        "positive": "Consistent on-time payment history.",
    },

    # Add one entry per feature.
    # `suppress: True` hides policy-masked features.
}
```

Features without an entry still work, but they receive generic fallback messages. CompileML measures reason coverage, records it in the artifact, warns when coverage is incomplete, and can make full coverage a validation requirement.

The tooling should not quietly pretend that generic feature names are suitable adverse-action notices.

## Seeing what the model did

The visualization package draws the outputs emitted by `decide()`. It does not independently recompute the model or explanation.

That matters: the chart cannot disagree with the deployed decision because both come from the same payload.

```python
from compileml.viz import (
    waterfall,
    decision_drivers,
    band_conditioned_decision_drivers,
    band_ladder,
)

waterfall(
    decide(artifact, row, include_contributions=True)
)

decision_drivers(sample_decisions, y=y_sample)
band_conditioned_decision_drivers(sample_decisions, y=y_sample)
band_ladder(score_decisions, y_sample)
```

The image at the top of this README was rendered by `waterfall_svg()` from the repository's committed reference artifact. That renderer also uses only the standard library.

## Validation

Run the validation framework against a holdout set:

```bash
compileml validate artifact.json \
  --csv holdout.csv \
  --y-col DEFAULT
```

It checks:

1. artifact integrity;
2. explanation reconciliation;
3. fidelity to the source model;
4. band coverage, score resolution, and band efficiency;
5. bad-rate monotonicity;
6. band-ladder churn;
7. explanation stability;
8. reason-code coverage.

These checks run against the compiled artifact through the same runtime used for production decisions. There is no separate notebook implementation allowed to become "almost the same" over time.

The command exits with `0` or `1`, so it can gate deployment in CI.

## Install

```bash
pip install compileml
```

Optional teacher integrations:

```bash
pip install compileml[xgboost]
pip install compileml[lightgbm]
```

Visualization dependencies:

```bash
pip install compileml[viz]
```

The compile side depends on NumPy and scikit-learn. The `compileml.runtime` package uses only the Python standard library.

If necessary, the runtime directory can be vendored into a constrained environment:

```text
src/compileml/runtime/
```

## Documentation

* [Quickstart](docs/quickstart.md)
* [FAQ](docs/faq.md)
* [Tuning the compilation](docs/howto/tuning.md)
* [Artifact specification](docs/ARTIFACT_SPEC.md)
* [Reason codes](docs/howto/reason-codes.md)
* [Recalibration without band churn](docs/howto/recalibrate.md)
* [Deploying to Python, SQL, and COBOL](docs/howto/deploy.md)
* [Validation framework](docs/howto/validate.md)
* [Visualization](docs/howto/visualize.md)
* [Executable notebooks](examples/)

## Roadmap

The current priorities are:

* compute exact attribution at leaf time — making explain-everything p-independent, cheap for full-book batch, and enabling reason-code emission in the SQL export;
* add Java and C exporters;
* complete calibrated-PD output in COBOL;
* add an optional NumPy batch scorer;
* add a fairness-audit module.

## License

Apache-2.0.

Copyright 2026 Carlos Ortiz.
