# Changelog

All notable changes to CompileML are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/). The artifact schema is versioned independently
inside each artifact (`schema_version`).

## [Unreleased]

### Added
- Decision artifact schema v2: integer-quantized leaves, integer calibration
  tables, fixed-point band ladders, half-micro exact attribution with the
  reconciliation identity, missing-value policy, SHA-256 verify-on-load.
- Pure-standard-library runtime (`compileml.runtime`): score, band, calibrate,
  explain — enforced stdlib-only by test.
- Compile side: teacher→whitebox distillation, tree extraction for
  scikit-learn / XGBoost / LightGBM (float32 input-precision handling for
  XGBoost), leaf quantization, artifact builder with reason-coverage warnings
  and reproducible (timestamp-free) builds.
- Band builders: `quantile_bands`, `monotone_quantile_bands`, and the
  search-and-certify `semantic_bands` / `governance_bands`.
- Zero-churn recalibration with predecessor-hash provenance chains.
- Eight-check validation framework running entirely against the artifact
  through the production runtime; CLI exit-code gating.
- Exports: SQL (full pipeline; engine-executed parity tests) and COBOL
  (score + band; artifact integers verbatim).
- `compileml` CLI: compile, inspect, verify, score, validate, export.
- Benchmarks reproducing every README number; docs site; executable examples.
- Visualization extra (`compileml[viz]`): payload-driven `waterfall`,
  `decision_drivers`, `band_drivers`, and `band_ladder` (matplotlib), plus a
  dependency-free `waterfall_svg`. Plots draw the deployed integers and never
  recompute them; the waterfall's segments are tested against the spec §7.4
  identity exactly.
- Runtime self-check: on artifacts recording `exact_attribution`, `decide()`
  refuses to emit an explanation whose residual is nonzero — every explained
  production decision now re-proves attribution integrity in place.

### Changed
- Documentation doctrine: explain everything by default. Complete attribution
  on every decision is real-time for credit decisioning and makes portfolio
  analytics census-complete; the O(p²) cost is a full-book batch concern,
  which the leaf-time attribution roadmap item targets.
