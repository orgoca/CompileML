# Determinism

## The claim, precisely

**Identical input bytes + identical artifact ⇒ identical integer outputs** —
`latent_int`, band, `pd_ppm`, every `impact_int`, and every reason code — on
any conforming runtime, any hardware, any language.

## Why it holds

Floating-point nondeterminism comes from *rounding under accumulation*:
`(a + b) + c ≠ a + (b + c)` in floats, so summation order, vectorization, and
fused-multiply-add all matter. CompileML removes the accumulation entirely:

1. **Compile-time quantization.** Every leaf value becomes an integer once:
   `value_micro = rha(leaf × learning_rate × 10⁶)`. The float model is then
   discarded — the integer model *is* the model.
2. **Integer everything.** Scoring sums `value_micro` in int64 (associative —
   order cannot matter). Band edges are integers. The calibration table is
   integers, interpolated with an integer division formula
   ([spec §2.2](../ARTIFACT_SPEC.md)). Attribution is integer differences.
3. **The one float op left is comparison.** Tree routing evaluates
   `x <= threshold`. IEEE 754 comparison is exact — it involves no rounding
   and no arithmetic. Given the same input bytes, every machine routes every
   row identically.

For models whose source framework compares in float32 internally (XGBoost hist
trees), the artifact records `input_precision: "float32"` and every runtime
quantizes inputs to binary32 before comparing — so even that subtlety is
reproduced identically everywhere.

## What is *not* claimed

- **Upstream float pipelines.** If your feature pipeline produces different
  bytes on different systems, decisions can differ. Producing identical input
  bytes is the caller's contract; CompileML's contract starts at the feature
  vector.
- **Cross-artifact equivalence.** Two artifacts compiled from the same model
  are two artifacts. The unit of governance is the hash.
- **Population-level stability.** Determinism says nothing about drift in who
  applies. It says your *measurement* of drift is exactly reproducible — any
  change in a monitored metric is attributable to the portfolio, never the
  tooling.

## How it's tested

- The SQL export executes in a real engine and must match the Python runtime
  integer-for-integer on every row.
- The COBOL export compiles under GnuCOBOL in CI and is diffed the same way.
- Rebuilding an artifact from identical inputs must reproduce the identical
  hash (no timestamps or randomness live in hashed content).
- CI runs the full suite across operating systems and Python versions and
  compares artifact outputs byte-for-byte.
