# Audit a compiled model for fair lending

`compileml.fairness` runs a three-layer audit over decisions the artifact
already made. It produces evidence a validator can inspect. **It does not
certify compliance with ECOA, Regulation B, or anything else**, and it is
written so that it cannot be mistaken for doing so.

```python
from compileml.fairness import FairnessAudit
from compileml.runtime import decide

# include_contributions=True is required for sections 6 and 8
decisions = [decide(artifact, row, include_contributions=True) for row in X_test]

audit = FairnessAudit(
    decisions, y_test, protected,
    labels={1: "Male", 2: "Female"},
    threshold_int=500,          # your cutoff, in display-scale latent units
    artifact=artifact, X=X_test,
    protected_feature=None,     # name it only if the model actually uses it
)
audit.compute_all()
audit.print_summary()
audit.section(6)                # one section at a time
```

## Why three layers

The first two are commodity. `fairlearn` and `aif360` compute approval
ratios, score distributions, error rates and calibration by group, they do it
well, and if that is all you need then use them.

The third layer is the reason this module exists here. It asks how the model
behaves *locally* per group, and two of its sections are only answerable
because a compiled artifact carries exact attributions and real reason codes.

| Layer | Sections | Question |
|---|---|---|
| Outcome | 1–3 | What decisions were made? |
| Performance | 4–5 | How well does it predict per group? |
| Decision geometry | 6–11 | How does it behave locally? |

## Layer 1 — outcome

**§1 representation** — group sizes, observed bad rates with Wilson
intervals, and per-feature distribution divergence when you pass `X`. That
last part matters: a score gap fully explained by an input gap is a different
conversation from one that is not, and it is better to see that before
arguing about the model.

**§2 score distribution** — group means, KS, Wasserstein.

**§3 approval rates** — rate per group with intervals, and the adverse
impact ratio against the four-fifths window. The window is reported alongside
the ratio rather than collapsed into a pass/fail, because 0.81 and 1.00 are
not the same finding and should not print as though they were.

## Layer 2 — performance

**§4 calibration by group** — is the PD equally honest for everyone? A model
can pass every outcome test and still systematically over-predict risk for
one group. That is its own finding and it is invisible at Layer 1.

**§5 error rates** — FPR, FNR, precision, recall.

## Layer 3 — decision geometry

**§6 attribution disparity — the headline.** Decomposes the mean group score
gap by feature, exactly:

```
mean group gap (half-micro) : -163,311.812611
sum of per-feature gaps     : -163,311.812611
residual                    :  0.000e+00
```

Zero, not small. Because the artifact's attribution reconciles to the score
with no residual at depth ≤ 2, the per-feature contributions to a group gap
*sum to the gap*. A share above 100% is a genuine finding rather than an
error — one driver widens the gap further than observed while others offset
it — and that statement is unavailable from any decomposition whose parts do
not sum to the whole.

This is what turns "the ratio is 0.91" into "36.9% of the gap is `PAY_1`",
which is a remediation plan rather than an observation.

**§7 feature swing.** How far each feature *can* move a score, per group.
Not a gradient: a compiled artifact is piecewise constant, so an
infinitesimal perturbation returns zero almost everywhere and measures only
whether a split happened to fall nearby. Swing evaluates at the artifact's
own thresholds, which is exact — the score between two thresholds is constant
by construction.

**§8 attribution concentration.** How many drivers it takes to explain a
decision, as the top driver's share and an effective count, `exp(entropy)`.
It reads directly: *this group's decisions are explained by 5.1 features on
average, that group's by 4.6*. A decision resting on one dominant driver is
legible and easy to write a notice for; one spread thinly across six is
neither.

??? note "Why this is not a curvature measure"
    It started as one, which suits a smooth compiling function with real
    manifold structure where a Hessian means something. A depth-2 ensemble is
    piecewise constant — second derivatives vanish and finite differences
    measure noise.

    The second attempt was interaction share: main-effect points versus
    pairwise grid points, read off the scorecard. Exact, and useless on real
    artifacts. A depth-2 ensemble trained on the UCI panel produced **zero**
    main effects and forty-seven interaction grids, because no tree split on
    a single feature. The share is then 100% for everybody — a fact about the
    model, not about any group.

    Concentration varies by row, so it survives contact with real artifacts.
    `model_interaction_structure(artifact)` still reports the main/interaction
    split as model-level context.

**§9 boundary fragility.** Distance to the cutoff per group, and the share
within a small band of it. An adverse impact ratio is a snapshot; this is its
derivative. A group clustered near the boundary is one policy tweak away from
a disparate outcome even when today's ratio passes comfortably.

**§10 adverse-action reason parity.** The distribution of top reason codes
per group, with total variation between them.

This is regulatory rather than statistical — it concerns what applicants are
*told* — and it is the second place exactness matters. A SHAP-based audit
ranks feature importances and treats the ranking as a stand-in for reasons.
Here `reasons_negative` **is** the code the applicant would be sent, produced
by the same runtime that produced the decision. The audit is of disclosed
reasons, not of a proxy for them.

**§11 counterfactual.** Flip the protected attribute and measure band flips.
When the attribute is not a model input the section reports itself
inapplicable *positively* — "the model does not use it, so the test does not
apply" — rather than silently skipping. That is the desired outcome and it
belongs in the report.

## What it refuses to do

Section 6 raises rather than approximating, in two cases:

- **No contributions.** The default `decide()` payload carries `attribution`
  as a *method label* only. Per-feature impacts need
  `include_contributions=True`.
- **Attribution is not exact.** Above depth 2 the residual is nonzero, the
  decomposition would not sum to the gap, and the reason to prefer this over
  a general fairness library evaporates. Compile at depth ≤ 2, or read the
  outcome layers only.

Sections that lack their inputs are recorded in `audit.skipped` with the
reason, and `summary()` prints them. A missing section is never silent.

## No SHAP

Every metric derives from `decide()` payloads and the artifact. The module
adds no dependency beyond what CompileML already requires, and in §6 and §10
that is not a convenience but the substance.
