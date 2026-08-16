# Exact attribution

## The promise

Every decision decomposes into per-feature integer contributions that sum —
exactly, in integer arithmetic — to the decision:

```
2 · (score(x) − score(baseline))  ==  Σ contributions  +  residual
```

This is the **reconciliation identity** ([spec §7.4](../ARTIFACT_SPEC.md)). An
auditor doesn't have to trust the explanation; they re-add it. When the
whitebox has depth ≤ 2, the residual is exactly zero and the artifact records
`exact_attribution: true`.

## How it works

For each feature *j*, the runtime computes the **main effect**
`d_j = score(x) − score(x with feature j at baseline)` and, for each pair
*(i, j)*, the **pairwise interaction**
`I_ij = score(x) − score(x₋ᵢ) − score(x₋ⱼ) + score(x₋ᵢ₋ⱼ)`.
The contribution is the Shapley-consistent allocation
`c_j = d_j − ½ Σᵢ I_ij` — each feature keeps its main effect and half of every
interaction it participates in.

Two integer subtleties make this audit-grade rather than approximately true:

- **Half-micro units.** The ½ would break integrality, so contributions are
  carried as `c2_j = 2·d_j − ΣI` — integers, exactly.
- **Largest-remainder display rounding.** Display-scale impacts are floor-divided
  and the missing units are handed to the largest remainders (deterministic
  tie-break), so the displayed integers also sum exactly to the displayed score.

## Why depth 2 matters

A depth-2 tree touches at most two features per path, so the model contains no
interactions of order ≥ 3 and the pairwise decomposition is *complete* — the
residual is zero as an arithmetic identity. Deeper whiteboxes are allowed
(builds warn; the artifact records the measured depth and sets
`exact_attribution: false`), and the residual becomes a real, reported number.
For consumer-facing reason codes, keep depth ≤ 2: reasons that don't sum to
the decision are not reasons.

## Cost, honestly

Exact pairwise attribution costs `1 + p + p(p−1)/2` ensemble traversals per
row — 300 traversals at 23 features (≈8 ms in the pure-Python runtime),
5,000+ at 100 features. It is exact, not sampled, and priced accordingly.

**Explain everything.** For live decisioning, single-digit milliseconds is
real-time — a credit decision's end-to-end budget contains bureau pulls
measured in hundreds of milliseconds, so the explanation is statistically
invisible. What explain-everything buys is structural: the explanation is
part of the decision record (computed at decision time, under the artifact's
hash, not reconstructed later); there is one payload shape instead of a
bifurcated score/explain path; and portfolio questions — marginal-band
composition, driver drift, fairness cuts — become census facts over complete
attributions rather than sample estimates with selection effects. This is why
`decide()` defaults to `explain=True`.

Where the quadratic cost is real: re-explaining an entire book in batch
(10M accounts × 8 ms ≈ 23 CPU-hours, growing with p²) and very wide feature
sets. The leaf-time exact algorithm on the roadmap addresses exactly that —
per-tree decomposition makes the cost independent of feature count — and
additionally enables reason-code emission in the SQL export. Live latency was
never the constraint.
