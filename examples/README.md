# CompileML examples

Five executable notebooks. Notebooks 01–03 and 05 are offline and
deterministic — they run top-to-bottom anywhere the package is installed.

CI executes 01–03 and 05 on every push. Notebook 04 is a rendered gallery: it opts
out of CI execution (`metadata.compileml.ci_execute = false`) and ships
committed figures instead, since every API it demonstrates is already covered
by `tests/test_viz.py`. Re-run it after changing anything in `compileml.viz`.

| Notebook | What it shows |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | The full pipeline: teacher → whitebox → bands → reason codes → hashed artifact → stdlib-runtime decision → eight-check validation. Includes the reconciliation identity re-added by hand. |
| [`02_deploy_sql_cobol.ipynb`](02_deploy_sql_cobol.ipynb) | The same artifact as a generated SQL query — **executed in SQLite and diffed against the Python runtime, integer for integer** — and as a generated COBOL program. |
| [`03_governance.ipynb`](03_governance.ipynb) | Tamper refusal (edit one integer, loading fails), zero-churn recalibration with the predecessor-hash chain, and certified bands that find five real risk plateaus but refuse to segment noise. |
| [`04_visualization.ipynb`](04_visualization.ipynb) | Reference gallery for `compileml[viz]`: waterfall arrow geometry, exact remainder truncation, the depth>2 residual bar, dependency-free SVG, every `color_by` / `value_color` / `sort_metric` option, per-band facets, and restyling with your own palette. |
| [`05_fairness.ipynb`](05_fairness.ipynb) | The three-layer fair-lending audit: adverse impact, calibration and error rates by group, then the decision-geometry layer — **a group score gap decomposed by feature with a zero residual**, adverse-action reason parity, boundary fragility, and the counterfactual flip. Uses the UCI credit-default panel where available, so the protected attribute is a real one. |

Notebooks 01 and 03 render their figures with
[`compileml.viz`](../docs/howto/visualize.md) (`pip install compileml[viz]`) —
waterfalls and driver plots drawn from the decision payloads themselves.

Notebook 05 prefers the UCI credit-default panel, since a fair-lending
example is worth little with an invented protected attribute. It falls back
to synthetic data when that is unavailable — CI runs offline — and prints
which source it used, so the numbers below it are never ambiguous.

To adapt to your own data: replace the synthetic generator cell with your
feature matrix, keep everything else. The one piece you must author yourself
is the [reason dictionary](../docs/howto/reason-codes.md).
