# CompileML examples

Three executable notebooks, all offline and deterministic (synthetic
credit-style data with a fixed seed) — they run top-to-bottom anywhere the
package is installed, and CI executes them.

| Notebook | What it shows |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | The full pipeline: teacher → whitebox → bands → reason codes → hashed artifact → stdlib-runtime decision → eight-check validation. Includes the reconciliation identity re-added by hand. |
| [`02_deploy_sql_cobol.ipynb`](02_deploy_sql_cobol.ipynb) | The same artifact as a generated SQL query — **executed in SQLite and diffed against the Python runtime, integer for integer** — and as a generated COBOL program. |
| [`03_governance.ipynb`](03_governance.ipynb) | Tamper refusal (edit one integer, loading fails), zero-churn recalibration with the predecessor-hash chain, and certified bands that find five real risk plateaus but refuse to segment noise. |

Notebooks 01 and 03 render their figures with
[`compileml.viz`](../docs/howto/visualize.md) (`pip install compileml[viz]`) —
waterfalls and driver plots drawn from the decision payloads themselves.

To adapt to your own data: replace the synthetic generator cell with your
feature matrix, keep everything else. The one piece you must author yourself
is the [reason dictionary](../docs/howto/reason-codes.md).
