# Deploying an artifact

One artifact, three execution surfaces — all producing identical integers.

## 1. Python runtime (any environment with Python ≥ 3.10)

`compileml.runtime` imports only the standard library — a property enforced by
a unit test that parses every runtime module's imports. That means:

- **Minimal serverless images.** A function handler needs `json`, `bisect`,
  and the artifact — no numpy, no scikit-learn, no model server.
- **Vendoring works.** Copying `src/compileml/runtime/` into a constrained
  codebase gives you the full decision path with zero dependency review.

```python
from compileml.runtime import load_artifact, decide

ARTIFACT = load_artifact("decision.json")        # verify=True by default

def handler(event, context):
    return decide(ARTIFACT, event["features"], explain=event.get("explain", True))
```

Operational note: explain everything — the default. A fully explained
decision costs under a millisecond, and because attribution is
aggregated per tree that cost does not grow with feature count. It is
real-time against any credit-decisioning SLA, and it keeps one payload shape
flowing through downstream systems with the explanation stored as part of the
decision record. `decide(…, explain=False)` remains available as the faster
score-only path for bulk pre-screening where no decision is communicated to a
customer. Full-book batch re-explanation, once the expensive case, is now
linear in trees; see the
[benchmark](https://github.com/orgoca/CompileML/blob/main/benchmarks/results.json)
for measured figures.

Missing values follow the artifact's `missing_policy` — `"baseline"` re-applies
the training-time imputation at decision time; `"reject"` refuses the row. NaN
never routes silently through a tree.

## 2. SQL (warehouses, batch decisioning)

```bash
compileml export decision.json --target sql --dialect ansi --out scorer.sql
```

One generated query: nested CASE-WHEN trees accumulating the artifact's
integers, clamp, display conversion, band ladder, and the calibrated PD in ppm.
Requirements: IEEE-754 double columns and 64-bit integer division truncating
toward zero — satisfied by PostgreSQL, SQLite, DuckDB, BigQuery, and peers.

The test suite executes the generated SQL in a real engine and asserts
**integer equality with the Python runtime on every row** — `raw_micro`,
`latent_int`, `band`, `pd_ppm`.

Impute before the query (per the artifact's baseline); SQL `NULL` comparisons
would silently skip branches, so the exporter's contract is non-null inputs.

## 3. COBOL (mainframes, core banking)

```bash
compileml export decision.json --target cobol --program-id CMLSCORE --out scorer.cob
```

A self-contained `>>SOURCE FORMAT FREE` program: the artifact's leaf integers
verbatim (`ADD 21077 TO F-ACCUM-MICRO`), the spec's integer division formula in
`COMPUTE`, a strict-`<` `EVALUATE` band ladder, and the calibration table as a
second `EVALUATE` that leaves the calibrated PD in `F-PD-PPM`. Compiles under
GnuCOBOL and Enterprise COBOL 6+; CI compiles the export and checks latent, band
and PD against the Python runtime row by row, for every calibration mode.

- Feature inputs are `COMP-2` (IEEE binary64). For decimal-arithmetic targets,
  compile the artifact with `build_artifact(threshold_decimals=…)` so every
  runtime — Python included — compares the identical quantized thresholds
  ([spec §11](../ARTIFACT_SPEC.md)).
- Scope: score, band and calibrated PD by default.

### Reason codes on the mainframe

```bash
compileml export decision.json --target cobol --explain --top-k 4 --out scorer.cob
```

`--explain` (`export_cobol(..., explain=True)`) adds exact attribution: after
scoring, the program leaves the top adverse and favorable reasons in
`REASON-NEG-CODE(i)` / `REASON-NEG-IMPACT(i)` and `REASON-POS-CODE(i)` /
`REASON-POS-IMPACT(i)`, with `REASON-NEG-COUNT` and `REASON-POS-COUNT` saying how
many slots are filled. The codes and display-scale integer impacts are the ones
`decide(..., explain=True)` returns, in the same order; CI compiles the program
under GnuCOBOL and checks them row by row, including forced ties.

It emits **codes and impacts, not message text**. The customer-facing wording
belongs to the institution's letter templates, keyed by code; carrying long
localized strings through `PIC X` fields would add weight without adding
anything the templates don't already do.

Each tree's features and the artifact's baseline are known when the program is
generated, so every comparison against a baseline value is resolved at export:
a tree's subset walks become short, fixed `IF` trees over the real inputs. The
program grows with tree count, which is why this is opt-in.

When the exporter refuses, it raises `ExportError` with a stable `code`, and the
CLI prints it and exits with status 2:

| Code | Cause | What to do |
|---|---|---|
| `EXPLAIN_NOT_EXACT` | The artifact's attribution is not exact (whitebox depth > 2), so reasons would not reconcile to the score. | Compile at depth ≤ 2, or export without `--explain`. |
| `REASON_CODE_NOT_ASCII` | A reason code — from the dictionary, or the `NEGATIVE_<name>` fallback — is not printable ASCII, which mainframe character sets would not carry unchanged. | Give that feature an ASCII `code` in the reason dictionary, or suppress it. |

The SQL export does not emit reason codes yet.

## Which surface for what

| Surface | Returns | Typical role |
|---|---|---|
| runtime `decide(explain=True)` | band, PD, exact reasons | decisioning API, adverse-action notices |
| runtime `decide(explain=False)` | band, PD, latent | bulk pre-screening (no customer-facing decision) |
| SQL export | band, PD, latent per row | warehouse batch, portfolio re-score |
| COBOL export | band, PD, latent; reason codes and impacts with `--explain` | core-banking / mainframe rails |

Whatever the surface, the integers agree — that's the point.
