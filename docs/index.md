# CompileML

**Compile tree-ensemble models into deterministic, auditable decision artifacts.**

CompileML separates training a model from running a decision. Train with full
flexibility, then compile the *decision* — score, calibrated probability, risk
bands, and reason codes — into one hashed JSON artifact that produces the same
integers on a Python laptop, in a SQL warehouse, and in a generated COBOL
program.

It began as a credit-risk tool, and the examples throughout these docs come
from credit. The same primitives apply wherever a model makes a decision about
an individual case and has to explain and reproduce it — fraud, insurance,
eligibility, clinical decision support. [Where it fits](concepts/where-it-fits.md)
maps the vocabulary and says where the fit is weaker.

```
model  ──compile──►  decision.json  ──runs on──►  stdlib Python │ SQL │ COBOL
                     (hashed, versioned)          same integers everywhere
```

## Where to start

- New here → [Quickstart](quickstart.md)
- Not in lending? → [Where it fits](concepts/where-it-fits.md)
- Want the exact rules → [Artifact specification](ARTIFACT_SPEC.md)
- "Why should I believe the determinism claim?" → [Determinism](concepts/determinism.md)
- Writing reason codes — adverse-action notices, alert reasons → [Reason codes](howto/reason-codes.md)
- Model risk / validation team → [Validate before deploying](howto/validate.md)

## The design in one paragraph

At compile time every leaf of the tree ensemble is quantized once to an integer
(micro-units, default 10⁻⁶). After that, scoring is integer addition, band
assignment is integer comparison, probability calibration is an integer table
lookup, and attribution is integer subtraction. The single remaining
floating-point operation is the split comparison `x <= threshold`, which IEEE
754 evaluates exactly — no rounding, no accumulation, no platform variance.
Determinism stops being a promise and becomes an arithmetic property, testable
by executing the same artifact in two engines and diffing integers.
