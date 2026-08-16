# FAQ

### How many trees should the whitebox have? What depth?

Measured answers beat rules of thumb: run
[`sweep_whitebox`](howto/tuning.md) on a holdout and pick the elbow of the
retention curve. The short version: trees are a cheap linear knob that never
costs you a guarantee; depth is the knob that takes exact attribution away
above 2. **Spend on trees, be stingy with depth.**

### Why is depth 2 the default?

It is the boundary of two guarantees at once. At depth ≤ 2 the pairwise
attribution is *complete* — the residual is zero as an arithmetic identity —
and the artifact collapses into an [exact scorecard](howto/tuning.md#producing-a-scorecard).
At depth 3 both break, for the same mathematical reason: three-way structure
appears. The tooling warns, records, draws, and refuses accordingly.

### If I keep adding capacity, don't I just get the teacher back and lose the point of compiling?

No. Fidelity converges toward the teacher; the compilation properties —
integer determinism, one hashed artifact, SQL/COBOL export, sub-millisecond
scoring — hold at any size. Only exact attribution is at risk, and only from
depth. What grows with trees is artifact bytes and explanation milliseconds,
linearly, and [`sweep_whitebox`](howto/tuning.md) prices both.

### How many bands should I use?

Either let the data answer — [`semantic_bands` / `governance_bands`](concepts/bands.md)
return the band count they can statistically *defend*, and honestly return
one band on noise — or sweep fixed K with `sweep_bands` and read the
retention-vs-K table.

### How do I know my banding isn't leaving money on the table?

[`band_efficiency`](howto/tuning.md#money-on-the-table-within-band-auc). The
`gini_gap` is the discrimination your ladder discards; per-band within-band
AUCs (with bootstrap CIs) tell you *where* — a band whose CI sits above 0.55
can still rank risk internally and is a refinement candidate. Validation
check 4 carries the same numbers on every run.

### Can I get a classic points scorecard out of this?

Yes — exactly, not approximately, at depth ≤ 2: `build_scorecard(artifact)`
or `compileml scorecard decision.json --format csv`. The printed tables
re-sum to every production decision bit-for-bit; a validator can reproduce
scores in a spreadsheet.

### Why not just use SHAP?

TreeSHAP is exact for trees and a fine analysis tool — the differences are
about *deployment*, not correctness. CompileML's explanation is computed on
the deployed object itself (not the pre-compilation model), in integer units
that re-sum to the decision, by a runtime with no ML dependencies, and it
travels into the SQL and COBOL exports. The explanation is part of the
decision record, under the artifact's hash, rather than a separate analysis
run that must be trusted to match.

### Can I compile my XGBoost classifier directly?

Directly compiled models must emit a latent in [0, 1] — a classifier's raw
margin lives in log-odds space and will be clamped into nonsense. Distill it:
`train_whitebox(X, model.predict_proba(X)[:, 1])`. Regressors on
probability-like targets compile directly, and the build warns when sample
latents fall outside range.

### What about neural networks?

Same route: any model that produces a probability-like latent can teach a
whitebox. The artifact never contains the network — it contains the distilled
trees, with the retention measured and recorded.

### What happens when I retrain or recalibrate?

Two mechanically distinct cases, distinguishable by hash. **Recalibration**
(`recalibrate_artifact`) refits the PD table on fresh outcomes while the
model and band edges stay byte-identical — provably zero band churn, with the
predecessor's hash recorded as a provenance chain. **Retraining** produces a
genuinely new artifact and a fresh governance cycle. See
[zero-churn recalibration](howto/recalibrate.md).

### How are missing values handled?

By declared policy inside the artifact: `"baseline"` re-applies the
training-time imputation at decision time; `"reject"` refuses the row.
NaN never routes silently through a tree comparison, in any runtime.

### Is the artifact hash a signature?

No — it is an integrity check. Loaders verify it by default and refuse a
tampered or corrupted document, but it does not prove *who* produced the
artifact. Provenance of authorship is your repository's and your process's
job.

### My semantic_bands returned one band. Is that a bug?

It is the honest answer: under your `eps_auc` strictness, the data cannot
statistically support discrete classes — either outcomes are too noisy or
the within-band separation you demanded isn't there. Loosen `eps_auc`
deliberately, or accept that band boundaries would be arbitrary. A banding
tool that always returns the requested K is a random number generator with
labels.

### Why is the full explanation slower than scoring?

Exact pairwise attribution costs `1 + p + p(p−1)/2` ensemble traversals —
single-digit milliseconds at typical feature counts, which is real-time for
credit decisioning ([explain everything](concepts/attribution.md#cost-honestly)
is the recommended default). The cost matters for full-book *batch*
re-explanation, which the leaf-time roadmap item addresses.

### Same artifact, same input — could two machines ever disagree?

Not within the contract: scoring is integer addition, banding integer
comparison, calibration integer lookup, and the single float operation
(`x <= threshold`) is exact under IEEE 754. CI proves it continuously —
committed reference integers replayed on three OSes, generated SQL executed
and diffed row-for-row, generated COBOL compiled and run. What is *not*
claimed: that your upstream feature pipeline produces identical bytes across
systems ([precisely stated](concepts/determinism.md)).
