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

Operational note: `decide(…, explain=False)` is the sub-millisecond hot path.
Full explanations cost O(features²) traversals — score everything, explain
declines and review queues.

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
`COMPUTE`, and a strict-`<` `EVALUATE` band ladder. Compiles under GnuCOBOL and
Enterprise COBOL 6+.

- Feature inputs are `COMP-2` (IEEE binary64). For decimal-arithmetic targets,
  compile the artifact with `build_artifact(threshold_decimals=…)` so every
  runtime — Python included — compares the identical quantized thresholds
  ([spec §11](../ARTIFACT_SPEC.md)).
- Scope: score + band (the mainframe decision path). Calibrated PDs and reason
  codes are runtime/SQL concerns; a COBOL calibration section is on the roadmap.

## Which surface for what

| Surface | Returns | Typical role |
|---|---|---|
| runtime `decide(explain=True)` | band, PD, exact reasons | decisioning API, adverse-action notices |
| runtime `decide(explain=False)` | band, PD, latent | high-volume scoring |
| SQL export | band, PD, latent per row | warehouse batch, portfolio re-score |
| COBOL export | band, latent | core-banking / mainframe rails |

Whatever the surface, the integers agree — that's the point.
