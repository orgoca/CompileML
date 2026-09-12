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

Exact pairwise attribution is aggregated **per tree**, which makes its cost
`O(trees)` and independent of feature count. Attribution is additive over
trees, and a tree responds only to the features it splits on — so a tree that
does not split on `j` contributes nothing to `j`, and the interaction term for
a pair vanishes unless the tree splits on **both**. Only each tree's own
features need enumerating.

A depth-2 tree splits on at most three features, so it is walked at most
2³ = 8 times however wide the model is. A perturbation derivation scores the
whole ensemble `2 + p + p(p−1)/2` times — the row, the baseline, each feature
perturbed, each pair perturbed.

From the committed benchmark, attribution alone on a 120-tree ensemble:

| features | perturbation | per tree | tree walks, perturbation | tree walks, per tree |
|---|---|---|---|---|
| 8 | 0.94 ms | 0.62 ms | 4,560 | 792 |
| 23 | 6.79 ms | 0.70 ms | 33,360 | 912 |
| 50 | 32.26 ms | 0.78 ms | 153,240 | 944 |
| 100 | 131.31 ms | 0.76 ms | 606,240 | 936 |

Flat, not merely faster. The walk counts are exact and hold on any machine;
the milliseconds are one laptop's. The benchmark's target spreads signal over
every feature, so at 100 features the trees genuinely split on 88 of them —
the flatness is not a model ignoring its inputs. At eight features the gain
is modest; the gap widens with the square of the feature count.

The integers are identical — integer addition is associative, so regrouping
the same sums is bit-identical rather than close, which is what lets the
committed determinism oracle police the change.

**Explain everything.** For live decisioning, under a millisecond is
real-time — a credit decision's end-to-end budget contains bureau pulls
measured in hundreds of milliseconds, so the explanation is statistically
invisible. What explain-everything buys is structural: the explanation is
part of the decision record (computed at decision time, under the artifact's
hash, not reconstructed later); there is one payload shape instead of a
bifurcated score/explain path; and portfolio questions — marginal-band
composition, driver drift, fairness cuts — become census facts over complete
attributions rather than sample estimates with selection effects. This is why
`decide()` defaults to `explain=True`.

Batch re-explanation used to be the place the cost hurt: 10M accounts at
8.4 ms is roughly 23 CPU-hours, and it grew with p². At 0.62 ms it is under
two, and it no longer grows with feature count. That is what makes
reason-code emission in the SQL export tractable rather than impractical.
Live latency was never the constraint.
