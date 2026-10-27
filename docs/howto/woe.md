# Weight-of-evidence inputs

A WOE-transformed column is an ordinary numeric column that takes one value
per bin. CompileML needs nothing special for it: compile the model on the WOE
values and the artifact scores WOE values. This page lists what changes when
the inputs look like that. Every statement here is pinned by
[`tests/test_woe_inputs.py`](https://github.com/orgoca/CompileML/blob/main/tests/test_woe_inputs.py),
which uses four WOE columns of three or four bins each, built from synthetic
data with a fixed seed. sklearn, XGBoost and LightGBM models all compile and
score within the usual quantization bound of the source model.

## What the artifact contains

The artifact holds split thresholds **in WOE units**. It does not hold the WOE
table, the raw attribute or the bin edges, so the transform stays in your
pipeline and must be the same one the model was trained on. In the sklearn
whitebox every threshold falls between two WOE values seen in training, so a
split separates bins and never cuts one.

If you refit the WOE table, the numbers the artifact sees change; compile a new
artifact rather than feeding new WOE values to the old one. To keep the link
visible, put the table's version in `metadata`, which is part of the hash:

```python
artifact = build_artifact(…, metadata={"woe_table": "2026-q3"})
```

## Bands and within-band AUC

Because every column takes few values, the latent score is a plateau: rows
with the same WOE values get the same latent and therefore the same band. A
band edge can sit between bin combinations but never inside one, so a band is
always a union of bin combinations.

The band builders run on a WOE latent without changes.
`governance_bands(latent, y, min_band_size=300)` on the test data returns
several bands and reports `band_auc` in its metadata; every band is at or
below the default cap of 0.60.

Not measured: whether WOE inputs give more or fewer certified bands than the
same model on raw continuous features, and how `semantic_bands` behaves on
them at its default strictness. [Risk bands](../concepts/bands.md) explains
why a strict `eps_auc` can leave a plateaued latent with few bands.

## Reason codes

Reasons are keyed by the column name, so the dictionary key is the WOE
column's name. The text should describe the underlying attribute, because
that is what the applicant understands. The WOE value is an internal encoding
and never appears in a notice:

```python
reasons = {
    "woe_util": {
        "code": "HIGH_UTILIZATION",
        "negative": "Credit utilization is high relative to available limits.",
        "positive": "Credit utilization is well managed.",
    },
    "woe_late": {
        "code": "LATE_PAYMENTS",
        "negative": "Payments were made after their due date.",
        "positive": "Payments were made on time.",
    },
}
display_names = {"woe_util": "Credit utilization", "woe_late": "Late payments"}

artifact = build_artifact(…, reasons=reasons, display_names=display_names)
```

A row in the lowest-WOE utilization bin then reports `HIGH_UTILIZATION` with
the `negative` sentence and the label `Credit utilization`. The rest of the
mechanism is in [Reason codes](reason-codes.md).

Two things follow from the inputs being bins:

- **The reason is about the bin.** The artifact never sees the raw value, so a
  reason says the applicant's bin scored worse than the baseline's bin, not
  where inside the bin they fall.
- **The baseline should be a bin.** Attribution is measured against the
  `baseline` row, and a row equal to the baseline has no reasons at all.
  `compile_selected` uses the per-column median of the development rows, which
  for a WOE column is a bin's value except at an exact tie, when the two middle
  rows sit in different bins and the median falls between them. If you call
  `build_artifact` directly, pass the WOE value of the bin you mean as the
  reference, such as the most common bin of each column.

## Monotone directions

Declared directions are in WOE units. With WOE defined as `ln(good/bad)`,
higher WOE means lower risk, so a model scoring risk is non-increasing in WOE
and the declaration is `-1`. If your WOE runs the other way, the sign flips.

```python
model, _ = train_whitebox(X_woe, y, monotone_constraints=[-1, -1, -1, -1])
artifact = build_artifact(…, monotone_constraints=[-1, -1, -1, -1])
```

The build re-verifies the compiled trees against the declaration, as
described in [Tuning](tuning.md#enforcing-monotone-directions). If the declaration
disagrees with the trees, the build raises. A wrong sign used for both training
and the declaration is not caught: it constrains the model against the data, and
you see it as lost discrimination. What the check guarantees is monotonicity in
the **WOE value**. Whether risk is monotone in the raw attribute is decided by how
you bin it.

## Missing values

If missing has its own WOE bin, give those rows that bin's WOE value before
they reach the artifact. It is then one more value of the column, and it is
scored and explained like any other bin.

Do not pass `NaN` to stand in for it. `train_whitebox` refuses `NaN`, and a
`NaN` reaching a compiled artifact is replaced by the baseline (under the
default `missing_policy`), which is the baseline bin and not the missing bin.
A `missing_as_bin` policy is an open design
([#11](https://github.com/orgoca/CompileML/issues/11)).

## Not yet tested

- Real portfolios. All of the above is checked on synthetic data.
- Whether WOE inputs change the band count compared with raw features, beyond
  the single case above.
- `compile_selected(..., reference="woe")` when `X` is already WOE-encoded. The
  reference fits its own WOE tables on whatever columns it is given.
- Monitoring and drift reports on WOE inputs. They run on any numeric input,
  but their behaviour on a column with few values is not covered here.
- Bins with very few rows, where the WOE value itself is unstable.
