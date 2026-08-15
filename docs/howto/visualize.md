# Visualize decisions

```bash
pip install compileml[viz]
```

One design rule governs `compileml.viz`: **plots draw decision payloads, they
never recompute them.** Every function consumes `decide()` output — the same
integers production emits — so a chart cannot disagree with the deployed
decision. The waterfall's bars sum to the score because the
[reconciliation identity](../concepts/attribution.md) says they must; the test
suite asserts it on the exact half-micro integers behind the plot.

## The waterfall — one decision, audited

```python
from compileml.runtime import decide
from compileml.viz import waterfall

decision = decide(artifact, applicant_row, include_contributions=True)
fig, ax = waterfall(decision, labels=DISPLAY_NAMES)
```

Baseline → per-feature impacts (largest first) → final score, with the band
and `latent_int` in the title. Features beyond `max_features` collapse into a
single balancing remainder bar; artifacts compiled deeper than depth 2 show
their interaction residual as an explicit bar rather than hiding it.

A validator can re-add the figure by hand. That is the point.

## Population views

```python
from compileml.viz import decision_drivers, band_conditioned_decision_drivers, band_ladder

sample = [decide(artifact, row, explain=True) for row in X_sample]
decision_drivers(sample, y=y_sample)                       # SHAP-style beeswarm by reason code
band_conditioned_decision_drivers(sample, y=y_sample)      # faceted beeswarm per band
band_ladder(score_only_decisions, y_sample)                # observed bad rate per band
```

The driver plots are the original beeswarm design: one point per
(decision, top-k reason), labeled by **reason code**, biggest drivers on top,
per-pile x-jitter to break discrete-value stacks, risk-increasing points drawn
slightly larger and on top. `color_by` selects the encoding — `"auto"` colors
by observed outcome when `y` is given and by impact direction otherwise — and
`value_color=True` / `value_alpha=True` (with `values=` one `{feature: value}`
dict per decision) add SHAP-style feature-value gradients toward dark low-end
counterpart colors.

Cost note: population driver plots need `explain=True` per sampled row —
O(features²) traversals each — so pass a *sample*, not the portfolio.
`band_ladder` is the exception: it only needs bands, so cheap
`decide(…, explain=False)` payloads suffice.

## Dependency-free SVG

```python
from compileml.viz import waterfall_svg

svg_text = waterfall_svg(decision)            # standard library only
open("decision.svg", "w").write(svg_text)
```

The payload is plain integers, so the waterfall renders without any plotting
stack — suitable for audit records, emails, and docs. Output is deterministic:
same payload, same bytes. The image on the project README is this function's
output, generated from the repo's committed reference artifact.

## Styling

Every renderer accepts `labels={feature: display_name}` and a `colors={…}`
override (keys: `up`, `down`, `base`, `remainder`, `residual`, `good`, `bad`,
`neutral`), and the matplotlib functions take an `ax=` to compose into your
own figures.
