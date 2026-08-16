# Tuning the compilation

Every question on this page has a measured answer — `compileml.tune` exists
so configuration is a table you read, not a guess you defend.

## The two knobs are not symmetric

| Knob | Buys | Costs | Determinism / portability |
|---|---|---|---|
| `n_estimators` ↑ | fidelity to the teacher | artifact size and explanation time, both **linear** | **unaffected** |
| `max_depth` ↑ | fidelity per tree | **exact attribution above 2**, per-tree explain cost, scorecard legibility | **unaffected** |

Integer exactness, cross-runtime determinism, and hash governance are
structural: a 500-tree artifact is exactly as deterministic and exactly as
COBOL-exportable as a 30-tree one. Nothing about compilation degrades with
size — what degrades above depth 2 is *explainability*, and only that.

**Spend on trees; be stingy with depth.**

## Finding the tree count

```python
from compileml.tune import sweep_whitebox

rows = sweep_whitebox(
    X_train, teacher_latent_train, y_train,
    trees_grid=(20, 40, 80, 160, 320),
    depth_grid=(1, 2),
    X_val=X_val, y_val=y_val, teacher_latent_val=teacher_latent_val,
)
# pandas.DataFrame(rows) if you like tables
```

Each row reports holdout Gini and retention versus the teacher, Spearman rank
agreement, `exact_attribution`, the quantized model's JSON size, and a
*measured* per-row exact-explanation cost. Read it like a cost curve: retention
climbs steeply, then plateaus; pick the elbow. In the repository benchmark,
120 trees at depth 2 retained 97.9% of a 300-tree teacher's Gini — beyond the
plateau you pay linear size and explain time for basis points of fidelity.

Always sweep on a holdout (`X_val=`) — in-sample retention flatters every
configuration, and the output flags `in_sample: true` when you didn't.

## Choosing depth: one table, one story

| depth | Attribution | Scorecard | Fidelity |
|---|---|---|---|
| 1 | exact, zero residual | **exact classic scorecard** (bin → points) | lowest |
| **2** (default) | exact, zero residual | scorecard + explicit interaction grids | good |
| 3+ | residual appears | none exists | highest |

Depth ≤ 2 is not a style preference — it is the boundary of two guarantees.
The pairwise decomposition is *complete* for functions with no three-way
interactions, which is precisely what depth ≤ 2 trees are; at depth 3 the
leftover becomes a real, reported residual
([why](../concepts/attribution.md)), and no clean scorecard exists for the
same mathematical reason. The tooling enforces the boundary honestly rather
than hiding it:

- `train_whitebox` **warns** at `max_depth > 2`;
- the artifact **records** `whitebox_max_depth` and `exact_attribution`;
- every explained decision **reports** its residual, and the waterfall draws
  it as an explicit bar;
- `decide()` **refuses** a nonzero residual on an artifact claiming exactness;
- `build_scorecard` **raises** above depth 2.

To *quantify* what depth 3 would cost you before committing, explain a sample
and look at the residuals directly:

```python
residuals = [
    abs(decide(artifact3, row, include_contributions=True)
        ["attribution_residual_half_micro"]) / (2 * 1_000_000)
    for row in X_sample
]
# share of decisions with any unexplained remainder, and how large it gets
```

## "Doesn't a bigger whitebox just become the teacher?"

Only in the sense you want. More capacity converges toward the teacher's
*predictions* while every compilation property — integer determinism, the
hashed single artifact, SQL/COBOL export, sub-millisecond scoring — holds at
any size. The one thing you can lose is exact attribution, and that is
controlled solely by depth, not trees. The trade to actually manage is
pragmatic: linear growth in artifact bytes and explanation milliseconds,
which `sweep_whitebox` prices per configuration.

## How many bands?

Two philosophies, both shipped:

**Discover K.** [`semantic_bands` and `governance_bands`](../concepts/bands.md)
return the number of bands the data can statistically *defend* — Jeffreys-CI
separation between neighbors, no residual rank power within any band,
bootstrap-certified. Feed them noise and they honestly return one band.

**Sweep fixed K.**

```python
from compileml.tune import sweep_bands

rows = sweep_bands(latent_train, y_train, k_grid=(4, 6, 8, 10, 12, 16))
```

Per K: band-ordinal Gini and retention, the Gini gap, the worst within-band
AUC with its verdict, minimum band volume, monotonicity violations, and any
integer-edge collisions (a K too fine for the display scale to represent —
the build would refuse it anyway).

## Money on the table: within-band AUC

A band ladder discards rank information by design; the governed question is
*how much* and *where*:

```python
from compileml.bands import band_efficiency

eff = band_efficiency(latent_val, y_val, artifact)
eff["gini_gap"]        # continuous Gini − band-ordinal Gini: the headline
eff["per_band"]        # n, bad rate, within-band AUC with bootstrap CI, verdict
eff["worst_band"]      # strongest refinement candidate
```

Reading the per-band verdicts:

- **exhausted** — CI upper bound ≤ 0.55: the score is used up inside this
  band; splitting it further separates noise, not risk.
- **refinable** — CI lower bound ≥ 0.55: the latent can still rank outcomes
  inside the band; a finer cut there would separate risk your policy
  currently treats as homogeneous. That is money on the table.
- **inconclusive** — the interval spans both stories; more volume before
  concluding anything.

The same diagnostics attach to every validation run: check 4 of
[`validate_artifact`](validate.md) reports `banding_gini_gap` and
`worst_within_band_auc` whenever outcomes are supplied, advisory by default
and gateable via `max_within_band_auc=` when your policy wants a hard limit.

## Producing a scorecard

At depth ≤ 2 the artifact *is* a points-based scorecard — exactly, not as an
approximation:

```python
from compileml.scorecard import build_scorecard, scorecard_to_markdown

scorecard = build_scorecard(artifact)
print(scorecard_to_markdown(scorecard, labels=DISPLAY_NAMES))
```

```bash
compileml scorecard decision.json --format csv --out scorecard.csv
```

Depth 1 collapses to the classic form — per feature, bin → points. Depth 2
adds explicit pairwise interaction grids over the union of the relevant
thresholds. Points are the artifact's own integers, and the identity

```
base_points + Σ main_effect(x) + Σ interaction(x) == raw_micro
```

holds bit-for-bit on every row (`score_from_scorecard` re-derives any
decision from the printed tables alone — the test suite asserts it). Hand
the CSV to a validator and they can reproduce production scores in a
spreadsheet.

Above depth 2, `build_scorecard` raises instead of approximating — the same
boundary as exact attribution, for the same reason.

## Defaults, for the impatient

`train_whitebox(n_estimators=30, max_depth=2)` and `n_bands=10` are sane
starting points, proven in the repository's own benchmark and examples. The
sweeps are for when "sane" needs to become "measured" — which, in a model
governance file, it eventually does.
