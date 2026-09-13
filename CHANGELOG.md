# Changelog

All notable changes to CompileML are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/). The artifact schema is versioned independently
inside each artifact (`schema_version`).

## [Unreleased]

### Added
- The COBOL export emits the calibrated PD
  ([#14](https://github.com/orgoca/CompileML/issues/14), part 1). The
  calibration table becomes an `EVALUATE` that leaves `F-PD-PPM` populated,
  with the same first-match rule and `div_rha` interpolation as the runtime
  and the SQL export, for `linear_int`, `step` and uncalibrated artifacts.
  The spec requires every exporter to reproduce §4–§6, and COBOL now does.
  The GnuCOBOL run-parity test checks latent, band and PD row by row in all
  three modes, including rows clamped at both ends of the score range.
- The COBOL export emits reason codes and integer impacts with
  `explain=True` (`compileml export --target cobol --explain`), completing
  [#14](https://github.com/orgoca/CompileML/issues/14). The program leaves the
  top `top_k` adverse and favorable codes and display-scale impacts in
  `REASON-NEG-*` / `REASON-POS-*`, matching `decide(..., explain=True)` exactly
  and in order. Each tree's features and the baseline are known at export, so
  every baseline comparison is resolved then and each subset walk is a short,
  fixed `IF` tree. Codes and impacts only: message text belongs to the
  institution's letter templates. Opt-in, because the program grows with tree
  count.

  Refusals raise `ExportError` with a stable code — `EXPLAIN_NOT_EXACT` for
  artifacts whose attribution is not exact, `REASON_CODE_NOT_ASCII` for codes
  a mainframe character set would not carry — and the CLI exits with status 2.
  GnuCOBOL parity covers fallback codes, a dictionary with an escaped quote and
  a suppressed feature, and an artifact built to force ties in both the
  largest-remainder rounding and the reason ranking.
- FAQ: *Why not PMML, ONNX, or m2cgen?*
  ([#17](https://github.com/orgoca/CompileML/issues/17)) — where each is the
  better choice, what CompileML trades for its guarantee (a distilled model,
  about 2% of Gini on the benchmark), and why the difference from PMML is
  integer arithmetic and ensemble attribution rather than reason codes, which
  PMML scorecards already carry.

### Fixed
- A feature whose sanitized COBOL name matched one of the program's own
  working-storage fields (for example `accum_micro` or `pd_ppm`) would have
  shadowed it. Those names are now reserved, and such a feature gets a
  suffix.
- The deploy guide said reason codes were a SQL concern; neither export
  emits them yet.
- The FAQ's SHAP answer said explanations travel into the SQL and COBOL
  exports. Neither emits reason codes yet (#14); it now says so.

## [0.6.0] - 2026-09-13

Stability monitoring, limited to what a compiled artifact makes possible.

A risk committee asks three things every period, and `compileml.monitor`
answers each from the decisions the artifact already produced: which
features moved the score — decomposed exactly, with a residual of zero;
whether the bands are still calibrated, read against the band each decision
logged; and whether the frozen baseline still describes the population.
It computes no PSI, CSI or distribution tests; the tools you already run do
those, and the guide shows how to feed them.

A minor release because `compileml.monitor` is new public API. Nothing else
changes behaviour: the runtime, exporters and artifact schema are untouched,
and fairness §6 now shares its arithmetic with drift decomposition, with
byte-identical output.

### Added
- `compileml.monitor` ([#9](https://github.com/orgoca/CompileML/issues/9)):
  stability monitoring limited to what a compiled artifact makes possible.
  - `drift_decomposition(reference_decisions, current_decisions)` splits the
    change in mean raw score, current minus reference, into per-feature
    shifts that sum to it with `==`. It reports each side's mean `latent_int`
    as context only, since the deployed latent is clamped.
  - `band_drift(artifact, decisions, y)` compares observed bad rates, with
    Wilson intervals, against the PD the artifact emitted, grouped by the
    band each decision logged.
  - `baseline_staleness(artifact, X_reference, X_current)` measures how far
    the frozen baseline has moved in population rank, with missing rates.

  No PSI, CSI or distribution tests: `docs/howto/monitor.md` explains how to
  feed existing tools from the logged bands, and why outcomes must have
  matured before band calibration means anything.
- `CODE_OF_CONDUCT.md`: the Contributor Covenant, version 2.1, unmodified
  apart from the reporting contact.

### Changed
- The exact gap arithmetic behind fairness §6 moved into one private module
  that `attribution_disparity` and `drift_decomposition` both call, so the
  float-means defect fixed in 0.5.2 cannot return through a second copy. §6
  output is byte-identical before and after on the UCI panel.
- `CONTRIBUTING.md` states the two-sides rule: the learning side may use any
  library; the inference side, where the artifact decides, is standard-library
  only and simple arithmetic, always. It links the Code of Conduct, and its
  starting points no longer list fair lending (#10), which shipped in 0.5.0.

## [0.5.2] - 2026-09-13

Fixes to what the fair-lending audit reports, and answers to an independent
evaluation. No change to the runtime, to scoring, or to any artifact a build
produces; the artifact schema is unchanged.

The fairness audit's §6 residual is now exactly zero rather than float noise
whose sign varied by machine, and §11 no longer claims a model ignores a
protected attribute it was simply not told about. The rest answers
[@deburky](https://github.com/deburky)'s evaluation of 0.4.3 on fraud data:
five usability observations addressed, and a README section recording what
that evaluation established and what it did not cover.

Additive only: `counterfactual` results gain a `status` key, and the
single-band builders gain a warning.

### Changed
- README: the roadmap still listed per-tree attribution and the fairness
  audit as future work, both shipped in 0.5.0. It adds an *Independent
  evaluation* section summarising
  [@deburky](https://github.com/deburky)'s test of 0.4.3 on fraud data,
  including what it did not cover, and links the fairness guide.
- The 0.5.0 entry was missing the single-band SQL fix (#40, #42); added.
- Docs set the depth-2 expectation: a depth-2 scorecard is mostly pairwise
  interaction grids, not one table per feature, and `max_depth=1` is the
  classic form. The fairness guide's grid count now matches its notebook
  (55, not 47).
- README notes that the XGBoost and LightGBM extras need `libomp` on macOS.

### Fixed
- `attribution_disparity` (§6) took group means in floats, so an exact
  decomposition reported a residual of about ±1e-11 whose sign depended on
  how the sums rounded — `print_summary()` could print `residual -0` under a
  guide promising "zero, not small". The means are now exact rationals over
  integer sums: the residual is `0.0`, and `mean_gap_half_micro ==
  sum_of_feature_gaps` holds with `==` regardless of machine or row order.
  The committed fairness notebook had itself been printing `1.455e-11`.

  The tests compared with `approx`, which is how this shipped. They now
  assert equality, and a regression test uses integer data on which float
  means provably leave a residual — checked first, so it cannot pass
  vacuously.

- `counterfactual` (§11) with `protected_feature=None` reported "None is not
  among the artifact's features, so the model cannot be using it directly".
  The audit cannot know that, and on the UCI panel it was false: `SEX` is a
  model input. `None` now reports that the protected attribute was not named
  as a model input and that this is not evidence the model ignores it, and
  names any column matching the attribute value for value. Results carry a
  `status` — `not_named`, `not_an_input`, `not_binary` or `flipped` — so the
  cases can be told apart without parsing prose.

- The fairness guide's §6 example was transcribed from a different run than
  the notebook it links to. It now shows the notebook's output.

- `build_artifact`'s out-of-range latent warning always said "distill
  margin-space models first (train_whitebox)" — circular advice for a model
  that came from `train_whitebox`. It now reads the range and names the
  likely cause: slight overshoot from a squared-error whitebox on a skewed
  target, which clamping handles and leaves the artifact valid, or a
  margin-scale spread, which needs distilling. The artifact is unchanged.
- `semantic_bands` and `governance_bands` returned a single certified band
  silently, which then dead-ended the quickstart flow. They now warn and
  point at `monotone_quantile_bands` and `min_band_size`.
  `flags.no_discrete_classes` remains the machine-readable signal; the hint
  stays out of metadata because metadata is hashed into the artifact.
- `recalibrate_artifact`'s docstring named `metadata.recalibrated_from`; the
  key it writes is `metadata.recalibration.recalibrated_from`. Documented as
  written rather than renamed, since renaming would break readers of existing
  artifacts.

  These five were reported by [@deburky](https://github.com/deburky) in notes
  on 0.4.3.

## [0.5.1] - 2026-09-12

Documentation only — no change to runtime code, no API or behaviour change.
Three docstrings in `src/` are corrected; nothing executable is. It exists
because two published surfaces only move on a release: the PyPI project page
renders the README at build time, and the documentation site pins its API
reference to a release tag so the docstrings it shows are the ones
`pip install` delivers. Both were still stating the cost of the attribution
path 0.5.0 replaced.

The committed benchmark records `"compileml": "0.5.0"`. That is accurate —
it was measured on 0.5.0 — and it was deliberately not re-run for the
version label: the runtime is identical between the two releases, and
re-measuring identical code would only move the timings by noise and make a
reviewed result look like a performance change.

### Fixed
- `benchmarks/results.json` had not been regenerated since `0.1.0-dev` and
  still described the perturbation attribution path. Re-run against 0.5.0: a
  fully explained decision is **0.62 ms** median, down from 8.4 ms.

  The benchmark now also times attribution alone at 8, 23, 50 and 100
  features by both derivations, and counts **tree walks per row** alongside
  the milliseconds. Walks are exact and machine-independent — 936 per row at
  100 features against 606,240 for the perturbation path — so the flatness
  claim no longer rests on one laptop's timings. It refuses to report a
  timing if the two derivations disagree on any integer.

  It records the CompileML version it measured. Running the benchmark against
  a stale installed copy had silently reproduced the old figure; it cannot
  pass for a measurement of the checked-out source any more.

  The retention block is unchanged. The recorded artifact hash changed
  because three metadata fields entered the hashed document —
  `bands.requested_n_bands` and `bands.scale` (#41, in 0.5.0) and
  `compileml_version` — while the model, band edges and calibration are
  bit-identical. Because the version string is hashed, that hash will move
  with every release; `rebuild_hash_identical` is the determinism signal.

  The 0.5.0 entry's cost table was measured on a 30-tree ensemble, where the
  benchmark uses 120. Per-tree cost scaling with tree count is the claim
  behaving as stated.

- Cost claims that #15 missed: the `decide()` docstring, the quickstart
  notebook, the deploy guide, and the README metrics table, which was
  hand-transcribed and contradicted its own prose. The perturbation path's
  traversal count is corrected to `2 + p + p(p−1)/2`, verified by
  instrumentation; it scores the baseline as well as the row.
- The README's examples list had dropped `05_fairness.ipynb`.

## [0.5.0] - 2026-09-12

Fair lending, and exact attribution that no longer costs what it used to.

The two land together for a reason. A fairness audit needs per-feature
contributions on **every** row, which made it the heaviest consumer of the
explanation path in the library — so the path got cheap first, and the
audit's numbers were written once rather than documented and revised.


### Changed
- Exact attribution is aggregated **per tree** instead of by perturbing one
  feature at a time ([#15](https://github.com/orgoca/CompileML/issues/15)).
  Attribution is additive over trees, and a tree responds only to the features
  it splits on — a tree that does not split on `j` contributes nothing to
  `j`'s main effect, and a pair's interaction term vanishes unless the tree
  splits on both. So only each tree's own features need enumerating, and the
  cost becomes `O(trees)`, **independent of feature count**:

  | features | perturbation | per tree |
  |---|---|---|
  | 8 | 0.26 ms | 0.124 ms |
  | 23 | 1.90 ms | 0.144 ms |
  | 50 | 8.84 ms | 0.128 ms |
  | 100 | 39.51 ms | 0.150 ms |

  Flat, not merely faster. **The integers do not change.** Integer addition is
  associative, so regrouping the same sums is bit-identical rather than close,
  and the committed determinism oracle — which pins every `impact_int` —
  passes unmodified.

  The perturbation derivation is kept as
  `contributions_half_micro_reference`, and validation check 2 now runs both
  and asserts they agree on every sampled row. Two derivations reaching the
  same integers is a stronger audit statement than one derivation asserting.

  Trees splitting on more than four distinct features fall back to the
  perturbation path, since enumerating subsets is exponential in a tree's own
  feature count — irrelevant at depth ≤ 2, where a tree touches at most three.

  The cost claims in the README, the attribution concept page and the FAQ were
  all true when written and are no longer; each has been rewritten rather than
  softened.

### Added
- `compileml.fairness` ([#10](https://github.com/orgoca/CompileML/issues/10)):
  a three-layer fair-lending audit over compiled decisions — outcome,
  performance, and decision geometry, eleven sections, computed from
  `decide()` payloads rather than by recomputing the model.

  Layers 1 and 2 are commodity and say so. The third is why the module lives
  here. **Attribution disparity** decomposes the mean group score gap by
  feature with a *zero* residual, because the artifact's attribution
  reconciles to the score exactly at depth ≤ 2 — a share above 100% is then a
  real finding rather than a rounding artefact. **Reason parity** audits the
  adverse-action codes an applicant would actually be sent rather than a
  feature ranking standing in for them. **Boundary fragility** reports who
  sits near the cutoff, which an adverse impact ratio cannot see.

  No SHAP, and no new dependency. Section 6 refuses rather than approximating
  when contributions are absent or the artifact's attribution is inexact.
  Sections lacking inputs are recorded with a reason rather than skipped
  silently, and nothing here certifies compliance with anything.

### Fixed
- `export_sql` emitted `CASE ELSE 'G01' END`, which is not valid SQL, for a
  single-band artifact; it now emits the label directly
  ([#40](https://github.com/orgoca/CompileML/issues/40), reported by
  [@deburky](https://github.com/deburky), fixed by
  [@tote10](https://github.com/tote10) in
  [#42](https://github.com/orgoca/CompileML/pull/42) — the project's first
  outside contribution). *This entry was missing when 0.5.0 was released and
  was added afterwards; the fix itself shipped in 0.5.0.*
- Quantile band builders no longer emit edges that `build_artifact` refuses
  ([#41](https://github.com/orgoca/CompileML/issues/41), reported and
  diagnosed by [@deburky](https://github.com/deburky) against real fraud
  data). `_quantile_edges` separated tied edges by `1e-12`, which satisfies a
  float strictly-increasing check and then collapses again at the display
  scale, so the ladder failed two steps later with an error naming the
  symptom rather than the cause. On a low base rate this is the ordinary
  case, not an edge one: a squared-error regressor distilled onto a skewed
  target clips a share of rows to exactly zero, and once that pinned mass
  exceeds about `1 / n_bands` of the volume a quantile cut necessarily lands
  inside it.

  `quantile_bands` and `monotone_quantile_bands` now take `scale=` and drop
  edges that would collide there, returning fewer bands than requested rather
  than an invalid ladder. The reduction warns and is recorded as
  `metadata["requested_n_bands"]`. The tuning guide explains the bound and
  points at the upstream cause.

## [0.4.3] - 2026-09-02

The first release since 0.4.0 whose tag, package version and contents agree.

Two releases were tagged one commit early, before the version bump had
merged. `v0.4.1` pointed at a tree still declaring 0.4.0, so it built a
duplicate and PyPI refused it. `v0.4.2` pointed at the 0.4.1 bump, so it
built and successfully published **compileml 0.4.1** — which is why 0.4.1 is
on PyPI and 0.4.2 never was. Both tags and their Zenodo archives are left
as they are: moving a published tag does not re-archive it, and rewriting
one others may have fetched is worse than a gap in the sequence.

### Fixed
- The release workflow compares the built wheel's version against the tag
  and fails in seconds if they disagree, instead of surfacing the mismatch
  at the upload step of an already-public release. `RELEASING.md` says to
  tag the *merge* commit when the bump goes through a pull request, and how
  to check it before publishing.

## [0.4.1] - 2026-09-02

Documentation and metadata only — no change to `src/`, no API or behaviour
change. It exists so the project has a citable, archived record: Zenodo
archives releases published after its GitHub integration is enabled, and
does not backfill, so the first release after switching it on is the first
one that gets a DOI.

### Added
- `CITATION.cff`. GitHub renders a "Cite this repository" button from it and
  Zenodo reads it when archiving, so the archived record carries the author,
  licence, abstract and keywords rather than what can be inferred from a
  tarball. The Zenodo concept DOI is not in this release: it is minted *by*
  the first archive, so the first archived version cannot contain it.
- `RELEASING.md` step keeping `CITATION.cff` in step with the version.
  `__version__` is the single source of truth for the package and everything
  builds from it, so an error there fails loudly; citation metadata is read
  only by GitHub and Zenodo, so a stale version there is silently published
  into a permanent archive.

### Fixed
- FAQ entries were `h3` under an `h1` with no `h2` between them, so the
  page's table of contents came out flat and the entries were invisible to
  it. They are now `h2`.

## [0.4.0] - 2026-08-30

The data the documentation compiles against, so a worked example can be run
rather than read.

### Added
- `compileml.datasets`: the data the documentation compiles against.
  `load_credit_default` fetches the UCI credit-card default panel once,
  verifies it against a recorded SHA-256 and caches it under
  `$COMPILEML_DATA_HOME` or `~/.compileml/data`; `make_credit_data`
  generates a synthetic set with a known interaction and needs no network.
  Demographic columns (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`) are excluded
  unless asked for — a credit library should not ship examples that model
  on them by default. The panel is fetched rather than bundled so the wheel
  stays small enough to vendor.

## [0.3.0] - 2026-08-30

Two yardsticks instead of one. Retention against a teacher measures the
distance to a ceiling and is structurally incapable of reporting that a
handful of logistic coefficients outscored the compiled whitebox. This
release adds the floor, and makes the training target something you
measure rather than assume.

Additive throughout: existing sweeps produce the same grid, every
positional call site is unchanged, and no artifact schema field moves.

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
