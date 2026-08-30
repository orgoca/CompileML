# Changelog

All notable changes to CompileML are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/). The artifact schema is versioned independently
inside each artifact (`schema_version`).

## [Unreleased]

### Added
- `compileml.reference`: a weight-of-evidence logistic regression used as a
  **floor**. `gini_retention_pct` measures distance to a teacher ceiling and
  is structurally incapable of reporting that a handful of logistic
  coefficients scored higher; this supplies the other side of the comparison.
  `fit_reference` bins each feature with a shallow supervised tree, encodes
  WOE, and picks the regularization strength by cross-validation with the
  tables refit inside every fold. It is a sanity floor, not a challenger
  model — `reference=` accepts a bare Gini float everywhere, so a champion
  scorecard's number can be used instead.
- `sweep_whitebox(reference=...)` adds `reference_gini`,
  `gini_vs_reference_pct` and `beats_reference` to every row, and now warns
  when it reports teacher retention with no floor beside it.
- Validation **check 10, reference floor**: the artifact must out-score a
  reference on the same data. Advisory by default, gated with
  `require_reference_floor=True` (the `max_within_band_auc` precedent).
- `sweep_whitebox(alpha_grid=...)` sweeps the training *target* across
  `alpha * y + (1 - alpha) * teacher_latent`, so distillation becomes a
  measured choice rather than an assumption. Defaults to `(0.0,)` — pure
  distillation, the existing behavior. Passing `teacher_latent=None` drops
  the teacher entirely and forces training on labels.

### Changed
- `train_whitebox`'s second parameter is now `target`, since labels and
  label/teacher blends are equally valid targets. `teacher_latent=` still
  works as a keyword and warns with `DeprecationWarning`; every positional
  call is unaffected.

## [0.2.1] - 2026-08-30

A correctness fix. Compiling a gradient boosting *classifier* produced an
artifact whose every score was short by the log-odds prior — invisible to
ranking, wrong in calibrated PD and band assignment. Such models are now
refused at extraction rather than compiled with a dropped base.

### Fixed
- **Gradient boosting classifiers compiled to silently wrong artifacts.**
  `extract_trees` read the initial estimator's constant only when it exposed
  `constant_`. A `GradientBoostingClassifier`'s init is a `DummyClassifier`,
  which does not, so `base` fell through to `0.0` and every compiled score was
  short by the log-odds prior. Because the error was a constant offset, Gini,
  Spearman and quantile band edges all agreed with the source model while
  calibrated PD and the fixed-point band ladder were wrong. Classifiers now
  raise with a pointer to `train_whitebox`, as `build_artifact` always
  documented but nothing enforced. A non-constant estimator `init` — where the
  compiled trees are only the residual and schema v2 has no field for the base
  model — raises for the same reason; `init="zero"` and the default constant
  init are unaffected.

  **Artifacts compiled from a classifier before this release carry a wrong
  base and should be rebuilt.** Ranking is unaffected, so a rank-only check
  will not reveal it; compare `pd_ppm` against the source model instead.
- The `sklearn` family no longer skips `validate_extraction`. It was exempted
  on the grounds of unit-test coverage, which existed for the regressor and
  not the classifier — the parity gate would otherwise have caught the above.

## [0.2.0] - 2026-08-27

Monotone feature constraints — the first of the three adoption-gap items
raised in review. A scorecard with a bin where more delinquency scores
*better* is one a committee rejects on sight; this release makes the
direction declarable, enforced at training, and — the part that matters for
governance — **verified against the compiled trees rather than promised by
the trainer**.

Additive and backward-compatible: `schema_version` stays 2, the new artifact
field is optional, and runtimes that predate it score constrained artifacts
to identical integers (verified against the published 0.1.1 runtime).

### Added
- Monotone feature constraints ([#8](https://github.com/orgoca/CompileML/issues/8)):
  `train_whitebox(monotone_constraints=...)` switches the whitebox backend to
  `HistGradientBoostingRegressor` (extraction parity tested; the
  unconstrained `GradientBoostingRegressor` path is byte-identical to
  before). `build_artifact(monotone_constraints=...)` verifies the declared
  directions against the compiled integer trees — refusing to build on any
  violation, independent of the trainer — and records them as
  `model.monotone_constraints` (hash-covered, spec §3.1). Validation gains
  check 9, re-verifying the declaration from the artifact alone; at depth
  ≤ 2, `scorecard_monotone_report` certifies the aggregate direction on the
  printed scorecard tables. `sweep_whitebox(monotone_constraints=...)`
  measures the monotonicity premium.

## [0.1.1] - 2026-08-16

First complete release. Supersedes 0.1.0, whose wheel carried a stale
hardcoded `compileml.__version__` of `0.1.0.dev0` (the PyPI metadata was
correct; the module attribute and artifact `compileml_version` were not).
The version now has a single source of truth — `compileml.__version__` —
which pyproject reads at build time.

### Added
- Tuning sweeps (`compileml.tune`): `sweep_whitebox` (trees × depth grid with
  measured retention, rank agreement, artifact size, and explanation cost)
  and `sweep_bands` (band-count grid with retention, Gini gap, and worst
  within-band AUC).
- `compileml.bands.band_efficiency`: the "money on the table" diagnostic —
  continuous-vs-band Gini gap plus per-band within-band AUC with bootstrap
  CIs and refinable / exhausted / inconclusive verdicts. Validation check 4
  now carries these fields whenever outcomes are supplied, advisory by
  default and gateable via `max_within_band_auc=`.
- Exact scorecard extraction (`compileml.scorecard`, stdlib-only): at
  whitebox depth ≤ 2 the artifact collapses into bin → points tables plus
  explicit pairwise interaction grids whose integers re-sum to every
  decision's `raw_micro` bit-for-bit (`score_from_scorecard` re-derives any
  decision from the printed tables; asserted in tests). Markdown and CSV
  renderers; `compileml scorecard` CLI subcommand; refuses above depth 2.
- Docs: tuning guide (`howto/tuning.md`) answering the configuration
  questions — tree count, depth, the trees-vs-depth asymmetry, band count,
  band efficiency, scorecards — and a FAQ page.
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
- `examples/04_visualization.ipynb`: rendered reference gallery for the viz
  suite — arrow geometry, exact remainder truncation, the depth>2 residual
  bar, dependency-free SVG, every colour/sort encoding, per-band facets, and
  restyling. Notebooks can now opt out of CI execution via
  `metadata.compileml.ci_execute = false`.

### Changed
- Documentation doctrine: explain everything by default. Complete attribution
  on every decision is real-time for credit decisioning and makes portfolio
  analytics census-complete; the O(p²) cost is a full-book batch concern,
  which the leaf-time attribution roadmap item targets.
