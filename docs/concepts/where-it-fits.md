# Where it fits

## Where it started

CompileML began as a credit-risk problem. A lender has to explain every
adverse decision to the applicant, reproduce any past decision for a
validator, and often run the decision on systems that predate Python. In that
setting the three properties this project is built around — the artifact
reproduces to the integer, it explains itself by arithmetic, and it lands where
the decision already runs — are not preferences but obligations. That is why
the examples, the datasets and much of the vocabulary here come from credit.

## Where it applies

Nothing in the mechanism knows about lending. A tree ensemble compiled into
integer arithmetic, attributions that add back to the score, a calibrated
probability, tiers, top drivers, a hash, and export to SQL are general.

The honest boundary is not "any machine learning". It is: **a model on tabular
data that makes a decision about an individual case, where the decision has to
be explained, reproduced exactly, audited, or run somewhere other than
Python.** Credit is the sharpest instance of that, not its definition. The
first independent evaluation of CompileML
([deburky/compileml-fraud-scoring](https://github.com/deburky/compileml-fraud-scoring))
went straight to fraud. The method carried over unchanged; of the two bugs it
found, one was a credit-shaped assumption — band builders that had never met a
base rate as low as fraud's.

| Domain | Fit | Why |
|---|---|---|
| Fraud and anti-money-laundering | strong | alert tiers, reasons an alert fired, warehouse scoring, audit trails |
| Insurance underwriting and claims | strong | regulated pricing, explanations owed to customers, legacy core systems |
| Eligibility and benefits screening | strong | decisions about individuals that must be explained and reproduced |
| Clinical decision support | strong, with care | why a case was flagged, reproducible scores; the field has its own regulatory frameworks, which this library does not address or certify |
| Churn and retention | moderate | warehouse scoring and per-customer drivers are useful; exact reproducibility matters less |
| Research on very wide data (e.g. genomics) | weak | thousands of features and usually no deployed individual decision — the fit returns when a validated score is deployed as a decision |

It fits poorly where the model has to stay deep to be useful (a depth-2
whitebox is the price of exact attribution), where inputs are images, text or
sequences rather than tabular features, or where no individual decision is
ever made and explained.

## Reading the vocabulary from another domain

The library's names come from where it started. They are kept as they are —
renaming them would break every artifact and integration built on them — so
read them this way:

| In CompileML | Meaning | In credit | In fraud | In clinical triage | In churn |
|---|---|---|---|---|---|
| positive outcome (`y = 1`) | the event the model predicts | default | fraud | adverse event | churn |
| `pd`, `pd_ppm` | calibrated probability of the positive outcome | probability of default | probability of fraud | risk of the event | probability of churn |
| band | an ordered tier of that probability | risk grade | alert tier | triage level | risk tier |
| cutoff range | where the decision threshold will sit | risk appetite | review capacity | escalation threshold | contact threshold |
| reason code | a top driver of one decision | adverse-action reason | why the alert fired | why the case was flagged | retention driver |
| `risk_increasing` / `risk_decreasing` | pushes toward / away from the positive outcome | raises / lowers default risk | raises / lowers fraud risk | raises / lowers event risk | raises / lowers churn risk |
| teacher retention | how much of the teacher's ranking survives compilation | same | same | same | same |

If you use CompileML outside credit, what compiling cost on your data is
exactly what [Show and tell](https://github.com/orgoca/CompileML/discussions/categories/show-and-tell)
is for.
