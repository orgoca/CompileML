# Reason codes

CompileML computes *which* features drove a decision and by *how much*. What a
customer reads is **your** content: only the institution can turn
`BILLS_PAID_LATE, impact +56` into a sentence that is accurate, compliant, and
humane. This page is the contract for supplying that content.

## The dictionary

One entry per feature, keyed exactly as the feature appears in
`features.names`:

```python
reasons = {
    "BILLS_PAID_LATE": {
        "code": "LATE_PAYMENTS",                                   # stable machine code
        "negative": "Recent payments were made after their due date.",
        "positive": "Consistent on-time payment history.",
    },
    "UTILIZATION": {
        "code": "HIGH_UTILIZATION",
        "negative": "Credit utilization is high relative to available limits.",
        "positive": "Credit utilization is well managed.",
    },
    "INTERNAL_SEGMENT_FLAG": {
        "code": "SEGMENT",
        "suppress": True,      # policy-masked: never shown as a reason,
    },                         # its contribution still exists and reconciles
}

artifact = build_artifact(…, reasons=reasons)
```

| Key | Used when | Notes |
|---|---|---|
| `code` | always | Stable identifier for downstream systems and adverse-action records. |
| `negative` | the feature is **risk-increasing** for this applicant | Customer-facing sentence. |
| `positive` | the feature is **risk-decreasing** | Customer-facing sentence. |
| `suppress` | `True` excludes the feature from reason lists | The contribution is still computed — reconciliation is never broken by suppression. |

Direction is determined per applicant by the sign of the contribution, so
every feature needs *both* sentences: high utilization hurts one applicant and
low utilization helps another.

## What happens without an entry

Nothing breaks — and that's the trap. Uncovered features fall back to
`NEGATIVE_<FEATURE>` codes and generic messages ("BILLS_PAID_LATE increased
the estimated risk"), which are mechanically correct and **not adverse-action
grade**. CompileML keeps this visible at every stage:

1. `build_artifact` **warns** with the exact list of uncovered features;
2. `metadata.reason_coverage` travels inside the hashed artifact;
3. `compileml validate --require-reasons` (or
   `validate_artifact(…, require_full_reason_coverage=True)`) turns coverage
   below 100% into a hard failure — recommended for any deployment that emits
   customer-facing notices.

## Writing good reason text

- Describe the *behavior*, not the model ("Recent payments were made after
  their due date", not "feature 12 exceeded its split threshold").
- Keep each sentence self-contained; downstream systems assemble notices from
  the top-k blocks.
- Legal/compliance review is part of the dictionary's lifecycle. The dictionary
  lives inside the hashed artifact, so a reviewed dictionary is a versioned,
  auditable object — changing a sentence changes the hash.

## How reasons are selected

Contributions are ranked by exact integer impact ([details](../concepts/attribution.md));
the top-k risk-increasing features become `reasons_negative`, the top-k
risk-decreasing become `reasons_positive`. Ties break toward the lower feature
index, deterministically. `top_k` defaults to the artifact's `runtime.top_k`
(5) and can be overridden per call.
