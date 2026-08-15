# CompileML

**Compile tree-ensemble models into deterministic, auditable decision artifacts.**

CompileML separates training a model from running a decision. Train with full
flexibility, then compile the *decision* — score, calibrated probability, risk
bands, and reason codes — into one hashed JSON artifact that produces the same
integers on a Python laptop, in a SQL warehouse, and in a generated COBOL
program.

```
model  ──compile──►  decision.json  ──runs on──►  stdlib Python │ SQL │ COBOL
                     (hashed, versioned)          same integers everywhere
```

## Where to start

- New here → [Quickstart](quickstart.md)
- Want the exact rules → [Artifact specification](ARTIFACT_SPEC.md)
- "Why should I believe the determinism claim?" → [Determinism](concepts/determinism.md)
- Writing adverse-action notices → [Reason codes](howto/reason-codes.md)
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
