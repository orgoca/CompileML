# CompileML Decision Artifact — Specification v2

**Status:** Draft (pre-0.1.0)
**Artifact type string:** `compileml.decision_artifact`
**Schema version:** `2`

This document is the normative contract for CompileML decision artifacts. Every
runtime (Python, COBOL, SQL, or any future target) MUST implement the algorithms
here exactly as written. If an implementation and this document disagree, the
implementation is wrong.

The keywords MUST, MUST NOT, SHOULD, and MAY are used as in RFC 2119.

---

## 1. Design goals

1. **Bit-identical decisions everywhere.** The same artifact and the same input
   bytes produce the same integer outputs on every platform, language, and
   hardware. This is achieved by doing *all arithmetic in integers*; the only
   floating-point operations permitted at decision time are **comparisons**
   (`x <= threshold`), which are exact under IEEE 754 and involve no rounding.
2. **Explanations that reconcile exactly.** Every decision decomposes into
   integer feature contributions that sum — exactly, in integer arithmetic —
   to the decision. The reconciliation identity (§7.4) is a testable invariant,
   not prose.
3. **One deployed truth.** Score, calibration, band ladder, feature contract,
   and reason codes travel in a single hash-verified JSON document. There is
   nothing else to version, and nothing to drift.

## 2. Integer conventions

All integers are signed 64-bit. Implementations MUST NOT overflow int64 for
artifacts within the limits in §10.

### 2.1 Rounding: half away from zero

The only rounding rule in this specification. For real `x`:

```
rha(x) = sign(x) * floor(|x| + 0.5)
```

### 2.2 Integer division with half-away rounding

For integers `num` and `den` with `den > 0`:

```
div_rha(num, den):
    if num >= 0:  return ( 2*num + den) // (2*den)
    else:         return -((-2*num + den) // (2*den))
```

This computes `rha(num / den)` using only integer operations. All targets
(Python `//`, COBOL `DIVIDE`, SQL integer division) MUST reproduce it exactly.

### 2.3 Scales

| Name | Field | Default | Used for |
|---|---|---|---|
| Display scale | `scale` | `1000` | `latent_int`, band `edges_int`, `impact_int` |
| Micro scale | `model.micro_scale` | `1000000` | leaf values, score accumulation, attribution |
| PD scale | fixed | `1000000` (ppm) | calibration output `pd_ppm` |

Constraint: `micro_scale % scale == 0`. Define `ratio = micro_scale / scale`.

Leaf values are quantized **once, at compile time**:

```
value_micro[leaf] = rha(leaf_value_float * learning_rate * micro_scale)
base_micro        = rha(base_float * micro_scale)
```

After compilation the floats are discarded. The integer model **is** the model:
parity, retention, and all downstream numbers are properties of the integer
model, not of the float model it came from.

## 3. Document layout

```jsonc
{
  "artifact_type": "compileml.decision_artifact",
  "schema_version": 2,
  "scale": 1000,
  "model": {
    "kind": "tree_ensemble_int",
    "micro_scale": 1000000,
    "base_micro": 285110,
    "n_features": 23,
    "input_precision": "float64",                   // "float64" | "float32" (§4)
    "trees": [
      {
        "feature":   [0, 3, -2, -2, 5, -2, -2],   // -2 = leaf sentinel
        "threshold": [1.5, 0.42, 0.0, 0.0, 2.5, 0.0, 0.0],
        "left":      [1, 2, -1, -1, 5, -1, -1],
        "right":     [4, 3, -1, -1, 6, -1, -1],
        "value_micro": [0, 0, -18342, 21077, 0, -4410, 33590]
      }
    ],
    "monotone_constraints": [1, 0, -1, ...]         // OPTIONAL (§3.1)
  },
  "calibration": {                                  // OPTIONAL (null allowed)
    "mode": "linear_int",                           // or "step"
    "f_micro": [1000, 52000, ...],                  // strictly increasing
    "pd_ppm":  [4100, 61200, ...]                   // non-decreasing
  },
  "bands": {
    "edges_int": [0, 11, 23, ..., 969],             // strictly increasing, display scale
    "labels": ["G01", "G02", ...],                  // len == len(edges_int) - 1
    "boundary": "left_closed_right_open"
  },
  "features": {
    "names": ["LIMIT_BAL", ...],                    // scoring order
    "baseline": [140000.0, ...],                    // float64; imputation + attribution reference
    "missing_policy": "baseline",                   // "baseline" | "reject"
    "display_names": {},                            // OPTIONAL
    "meta": []                                      // OPTIONAL
  },
  "reasons": {                                      // OPTIONAL reason dictionary
    "LIMIT_BAL": {
      "code": "LOW_LIMIT",
      "negative": "…",                              // shown when risk-increasing
      "positive": "…",                              // shown when risk-decreasing
      "suppress": false                             // OPTIONAL policy mask
    }
  },
  "runtime": {
    "attribution": "pairwise_interaction_int",
    "top_k": 5,
    "whitebox_max_depth": 2,                        // recorded; exactness depends on it
    "exact_attribution": true                       // true iff whitebox_max_depth <= 2
  },
  "metadata": {},                                   // free-form provenance
  "artifact_hash": "sha256 hex of §9"
}
```

`feature[i] == -2` marks node `i` as a leaf; its `value_micro[i]` is the leaf
payload. Interior nodes have `value_micro == 0`. Thresholds are float64 and
MUST round-trip exactly through JSON (shortest-repr serialization).

### 3.1 Monotone constraints (optional)

`model.monotone_constraints`, when present, is a list of length
`n_features` over `{-1, 0, +1}`: the declared direction of the compiled
score in each feature (+1 non-decreasing, −1 non-increasing, 0
unconstrained). The field is covered by the hash (§9) like everything
else.

The declaration is a *verified property of the shipped trees*, not a
training-time promise: builders MUST NOT emit the field unless the
quantized ensemble satisfies it (CompileML re-verifies tree-by-tree at
build and refuses otherwise), and validators re-verify it from the
artifact alone — validation check 9. Runtimes ignore the field; it
changes no decision, only what can be claimed about them. Absent field
means no directions are declared.

## 4. Scoring (normative)

Input: `x`, an array of float64 of length `n_features`, ordered by
`features.names`, after missing-value handling (§8).

```
score_micro(x):
    acc = base_micro
    for each tree:
        node = 0
        while feature[node] != -2:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        acc += value_micro[node]
    return acc                        // "raw" score, may fall outside [0, micro_scale]

latent_micro = clamp(score_micro(x), 0, micro_scale)
latent_int   = div_rha(latent_micro, ratio)
```

Notes:

- The float comparison `x <= threshold` is the **only** floating-point
  operation. Comparisons are exact: no rounding, no accumulation, no
  platform variance. Given identical input bytes, routing is identical
  everywhere.
- **Input precision.** If `model.input_precision` is `"float32"`, the runtime
  MUST round every input value to IEEE binary32 (round-to-nearest-even) before
  any comparison, then widen back to float64. This exists for models whose
  source framework compares in float32 internally (XGBoost hist trees): the
  compiler stores float32-adjusted thresholds, and quantizing the inputs makes
  every conforming runtime reproduce the source model's routing exactly.
  `"float64"` (the default, and always the case for distilled whiteboxes)
  means inputs are used as given.
- Accumulation order is irrelevant (integer addition is associative), so
  parallel or reordered implementations remain bit-identical.

## 5. Band assignment (normative)

Bands partition the display-scale latent line, left-closed right-open:
band `i` covers `[edges_int[i], edges_int[i+1])`; values below `edges_int[0]`
fall in band 0, values at or above `edges_int[-1]` fall in the last band.

```
band_index(latent_int):
    cutoffs = edges_int[1:-1]                 // interior edges only
    idx = count of c in cutoffs with c <= latent_int   // == bisect_right
    return idx                                // already in [0, n_bands-1]
```

This is the single band-assignment algorithm. Implementations MUST NOT provide
a parallel float-space path.

## 6. Calibration (normative)

Input: `latent_micro` (clamped). Output: `pd_ppm`, an integer in `[0, 1000000]`.
If `calibration` is null, `pd_ppm = div_rha(latent_micro * 1000000, micro_scale)`.

With `k = len(f_micro)` (compile-time guarantees: `f_micro` strictly
increasing, `pd_ppm` non-decreasing, `k >= 1`):

```
calibrate(latent_micro):
    if latent_micro <= f_micro[0]:    return pd_ppm[0]
    if latent_micro >= f_micro[k-1]:  return pd_ppm[k-1]
    hi = smallest index with f_micro[hi] > latent_micro
    lo = hi - 1
    if mode == "step":
        return pd_ppm[lo]
    // mode == "linear_int"
    num = (pd_ppm[hi] - pd_ppm[lo]) * (latent_micro - f_micro[lo])
    den = f_micro[hi] - f_micro[lo]
    return pd_ppm[lo] + div_rha(num, den)
```

Bounds: `num <= 10^6 * 10^6 = 10^12`, well inside int64.

## 7. Attribution (normative)

Attribution decomposes the **raw** (unclamped) score against the baseline row.
Let `S(v) = score_micro(v)`, `full = S(x)`, `fbase = S(baseline)`, and `n` the
number of features.

### 7.1 Main effects and pairwise interactions

```
for j in 0..n-1:
    d[j] = full - S(x with feature j set to baseline[j])          // integer micro

for i < j:
    I[i][j] = full - S(x_i→base) - S(x_j→base) + S(x_{i,j}→base)  // integer micro

Isum[j] = Σ_{i≠j} I[min(i,j)][max(i,j)]
```

### 7.2 Contributions in half-micro units

The Shapley-consistent allocation for a pairwise decomposition assigns each
feature its main effect plus **half** of each interaction it participates in:
`c_j = d_j − Isum_j / 2`. To keep this integral, contributions are carried in
**half-micro units** (denominator `2 * micro_scale`):

```
c2[j] = 2*d[j] - Isum[j]              // integer, exact
```

### 7.3 Residual

```
residual2 = 2*(full - fbase) - Σ c2[j]
```

For a model whose trees have depth ≤ 2, `residual2 == 0` **exactly** — this is
an arithmetic identity for functions with no interactions of order ≥ 3, and
because every quantity is an integer there is no floating-point caveat. For
deeper trees the residual is a real, reportable quantity. Compilers MUST record
`runtime.whitebox_max_depth` and set `exact_attribution = (depth <= 2)`;
validators SHOULD fail an artifact claiming exactness whose residual is nonzero.

### 7.4 Reconciliation identity (the auditable invariant)

```
2*(full - fbase)  ==  Σ c2[j]  +  residual2
```

Exact, in integers, always. When `exact_attribution` is true, additionally
`residual2 == 0`. This identity is what an auditor re-computes.

### 7.5 Display rounding (largest remainder)

Display contributions at the display scale must also sum exactly. Target sum
`T = div_rha(2*(full - fbase) - residual2, 2*ratio)`. Naive per-item rounding
of `c2[j] / (2*ratio)` can miss `T` by a few units, so allocate by largest
remainder:

```
q[j] = floor_div(c2[j], 2*ratio)            // floor toward -inf
r[j] = c2[j] - q[j]*(2*ratio)               // 0 <= r[j] < 2*ratio
deficit = T - Σ q[j]                        // 0 <= deficit <= n
give +1 to the `deficit` items with largest r[j];
ties broken by lower feature index first.
impact_int[j] = q[j] (+1 if selected)
```

Then `Σ impact_int == T` exactly, deterministically.

### 7.6 Reason dictionary and reason codes

**The reason dictionary is user-supplied content, not something CompileML can
invent.** The compiler knows that `BILLS_PAID_LATE` pushed a decision up or
down; only the institution knows how to say that to a customer. Human-readable
reason codes therefore work exactly as well as the dictionary you provide.

Each entry in the top-level `reasons` object maps a feature name (exactly as
it appears in `features.names`) to:

| Key | Required | Meaning |
|---|---|---|
| `code` | recommended | Stable machine code for the reason (e.g. `"LATE_PAYMENTS"`). Used in downstream systems and adverse-action records. |
| `negative` | recommended | Customer-facing message when the feature is **risk-increasing** for this applicant (e.g. *"Recent payments were made after their due date."*) |
| `positive` | recommended | Customer-facing message when the feature is **risk-decreasing** (e.g. *"Consistent on-time payment history."*) |
| `suppress` | optional | `true` excludes the feature from reason lists entirely (policy-masked features). Its contribution still exists and still reconciles. |

Selection and fallback rules:

- A contribution with `c2[j] > 0` is **risk-increasing** (adverse); `< 0` is
  risk-decreasing. Zero contributions produce no reason.
- Features whose reason entry has `suppress: true` are excluded from reason
  lists (they still appear in contributions).
- `reasons_negative` = top `top_k` adverse by `c2` descending;
  `reasons_positive` = top `top_k` favorable by `c2` ascending.
  Ties broken by lower feature index.
- Message selection: adverse → entry `negative`, favorable → entry `positive`.
- **Fallback:** a feature with no dictionary entry still produces a reason
  block, with code `NEGATIVE_<name>` / `POSITIVE_<name>` and a generic message
  naming the feature. Fallback output is mechanically correct but is **not**
  suitable for consumer-facing adverse-action notices.

Coverage requirements:

- Compilers MUST compute `reason_coverage` (fraction of `features.names` with
  a dictionary entry), record it in `metadata`, and warn at compile time when
  coverage is below 1.0, listing the uncovered features.
- Validators SHOULD surface reason coverage; deployments that emit
  consumer-facing notices SHOULD require coverage of 1.0 for all
  non-suppressed features.

## 8. Missing values

`features.missing_policy` is part of the contract:

- `"baseline"` (default): before scoring, any missing input (`null`, `NaN`)
  is replaced by `features.baseline[j]` — reproducing at decision time exactly
  the imputation the compiler saw at training time.
- `"reject"`: any missing input is an error; the runtime MUST refuse to score.

Runtimes MUST NOT silently route NaN through tree comparisons.

## 9. Hash

`artifact_hash` = SHA-256 (hex) of the canonical JSON serialization of the
document **without** the `artifact_hash` field: keys sorted lexicographically
at every level, separators `(",", ":")`, UTF-8, floats in shortest-repr form.
Loaders MUST verify the hash by default and refuse artifacts that fail.

## 10. Limits

| Quantity | Limit | Reason |
|---|---|---|
| `micro_scale` | ≤ 10^9 | keeps §6 products inside int64 |
| trees × max |leaf_micro| | ≤ 10^15 | accumulator headroom |
| features per artifact | ≤ 10^4 | attribution is O(n²) score calls |

## 11. Export parity

A conforming exporter (COBOL, SQL, Java, …) MUST emit logic that reproduces
§4–§6 bit-for-bit: the same `value_micro` integers, the same `div_rha`, the
same band cutoff rule. Because leaf values are integers in the artifact,
"round each leaf then sum" versus "sum then round" can no longer diverge —
there is nothing left to round. Threshold literals MUST be emitted with full
precision (shortest round-trip repr); if a target cannot represent a threshold
exactly, the artifact MUST be compiled with quantized thresholds so that every
target, including Python, sees identical values.

## 12. Determinism statement (what is and is not claimed)

**Claimed:** identical input bytes + identical artifact ⇒ identical integer
outputs (`latent_int`, `band`, `pd_ppm`, `impact_int[]`, reason codes) on any
conforming runtime, any hardware, any language. No environment, library
version, or accumulation order can change a decision.

**Not claimed:** that upstream feature pipelines produce identical float bytes
across systems (that is the caller's contract), or that two *different*
artifacts compiled from the same model are interchangeable — an artifact is
identified by its hash, and the hash is the unit of governance.

---

*Changelog*
- **v2, additive** — optional `model.monotone_constraints` (§3.1): declared,
  build-verified monotone directions. Absent field means unconstrained;
  `schema_version` unchanged.
- **v2** — integer-quantized leaves, integer calibration, half-micro exact
  attribution with largest-remainder display rounding, missing-value policy,
  hash-verified loads. Supersedes a pre-release float-scoring layout that
  quantized only at the output boundary.
