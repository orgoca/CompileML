# Monitor a deployed artifact

`compileml.monitor` answers the three questions a risk committee asks every
period, using only what a compiled artifact makes possible:

| Question | Function |
|---|---|
| Which features moved the score? | `drift_decomposition` |
| Are the bands still calibrated? | `band_drift` |
| Does the baseline still describe anyone? | `baseline_staleness` |

```python
from compileml.monitor import band_drift, baseline_staleness, drift_decomposition
from compileml.runtime import decide

reference = [decide(artifact, row, include_contributions=True) for row in X_reference]
current = [decide(artifact, row, include_contributions=True) for row in X_current]

drift = drift_decomposition(reference, current)
bands = band_drift(artifact, matured_decisions, matured_outcomes)
stale = baseline_staleness(artifact, X_reference, X_current)
```

It does not compute PSI, CSI or distribution tests, and it draws no
dashboards. The tools you already run do those well; see
[the last section](#psi-and-everything-else) for how to feed them.

## Which features moved the score

PSI says the score distribution moved. CSI says an input moved. Neither says
which feature *drove the score*. `drift_decomposition` does, exactly: the mean
raw score moved by Δ between the reference and current populations, and here
are per-feature shifts that sum to Δ.

The direction is always **current minus reference**. The report has the same
shape as the fairness audit's §6 — `mean_gap_half_micro`,
`sum_of_feature_gaps`, `residual`, `by_feature` — because it is the same
arithmetic: integer sums and exact rational means, so `residual` is `0.0` and
the parts equal the whole with `==`, on any machine and in any row order.

**What is decomposed is the raw score, not the deployed latent.** The runtime
clamps `latent_int` to the unit interval, and a clamped value does not
decompose. Each side's mean `latent_int` is reported under `latent_context`
for reading alongside; where many scores sit at the clamp, it will move less
than the raw score does.

**Read the shifts before the shares.** A share above 100% or below zero is a
real finding — one feature pushed the score further than it moved overall
while others pulled back. But when the total shift is small, every share is
large and unstable. The half-micro shifts are the stable numbers.

**Logs without contributions.** Production payloads often omit per-feature
contributions to save space. Re-run `decide(..., include_contributions=True)`
on the rows; attribution costs under a millisecond per row on the committed
benchmark, independent of feature count.

**After a recalibration.** `recalibrate_artifact` issues a new hash, and
`drift_decomposition` refuses decisions from two artifacts. The model and
baseline are unchanged, so re-run `decide()` on the reference rows under the
current artifact — the contributions come out identical.

**Missing values.** Under the `baseline` missing policy a missing value is
imputed at the baseline and contributes exactly zero. A jump in a feature's
missing rate therefore appears as that feature's contribution shrinking
toward zero. Read the decomposition next to `baseline_staleness`, which
reports missing rates.

## Are the bands still calibrated

`band_drift` compares, per band, the observed bad rate with the PD the
artifact emitted on those same decisions:

- rows are grouped by the band **each decision logged**, never by a band
  rebuilt from scores, so a row on a boundary counts where production put it;
- the observed rate carries a Wilson interval;
- `outside_interval` says whether the band's mean emitted PD falls outside
  that interval, and is `None` below `min_n` (default 30), where a flag would
  mean little;
- every band in the artifact is listed, including empty ones.

**Outcomes must have matured.** Use a vintage whose performance window has
closed. Last month's decisions have not had time to default and will look
better calibrated than they are. `drift_decomposition` needs no outcomes, so
it can run on this week's applications; `band_drift` cannot.

**Expect some flags from a calibrated artifact.** At 95% intervals each band
has about a one-in-twenty chance of flagging by sampling noise alone. With ten
bands, a perfectly calibrated artifact flags about one band every two periods
on average. A band that flags period after period is the signal.

Nothing is recalibrated automatically. A persistent miscalibration is the
evidence for an explicit call to
[`recalibrate_artifact`](recalibrate.md), which refits the PD table without
moving a single band edge.

## Does the baseline still describe anyone

The frozen baseline is both the value missing inputs are imputed with and the
reference point every attribution is measured from. When the population
moves away from it, imputation becomes biased *and* attributions are measured
against a point that no longer describes anyone.

Raw deltas are not comparable across features — a movement of 3 means one
thing for age and another for utilisation — so `baseline_staleness` measures
rank. For each feature it reports the baseline's percentile among the
non-missing values of each population, counting ties as half:

```
percentile = (count(x < baseline) + 0.5 * count(x == baseline)) / n_non_missing
```

and the shift between the two, alongside both missing rates. Features are
ranked by absolute shift. A baseline set at the development median starts near
0.5; a current percentile of 0.2 means the typical applicant now sits well
above it. No pass/fail threshold is applied: how much movement matters depends
on the feature.

This is deliberately narrower than CSI. It asks only whether the artifact's
own reference point is still representative, not whether a distribution
changed in general.

## PSI and everything else

Compute PSI with the monitoring tool you already run. The one thing to get
right is the bins: use the band each decision **logged**, not bins re-derived
from float scores. Rows sitting on a boundary can otherwise land in a
different bucket than production put them in, and the monitoring quietly
measures a different partition than the decisions did. Counting the logged
`band` field in each population gives you both distributions directly, and
needs no outcomes.

Because the artifact is hashed and its logic cannot change between runs, any
movement a monitor reports is movement in the population, not in the tooling.
Most monitoring setups cannot separate those two cleanly.
