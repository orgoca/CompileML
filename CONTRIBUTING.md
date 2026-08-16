# Contributing to CompileML

Thanks for your interest. CompileML aims for a small number of hard promises,
each enforced by tests — contributions are judged against that bar.

## Where to start

The [open issues](https://github.com/orgoca/CompileML/issues) are written to be
picked up: each one states the problem, why it matters in a regulated lending
context, a proposed approach, acceptance criteria, and the files involved.

Three of them are, in my view, what stands between this being interesting
infrastructure and something a risk function could actually adopt:

| | |
|---|---|
| [#8](https://github.com/orgoca/CompileML/issues/8) | **Monotone feature constraints.** Today the distilled whitebox cannot enforce them, so a compiled scorecard may show a bin where more delinquency scores *better*. That is a scorecard a committee rejects on sight. |
| [#9](https://github.com/orgoca/CompileML/issues/9) | **Stability monitoring (PSI/CSI/drift).** The validation framework checks an artifact at a point in time; model risk management is about what happens next. |
| [#10](https://github.com/orgoca/CompileML/issues/10) | **Fair lending.** Disparate impact testing, plus disparity decomposition over the exact attributions — something the reconciliation identity makes possible here in a way it is not elsewhere. |

Smaller entry points: [#17](https://github.com/orgoca/CompileML/issues/17)
(FAQ: why not PMML/ONNX), [#18](https://github.com/orgoca/CompileML/issues/18)
(WOE compatibility docs), [#12](https://github.com/orgoca/CompileML/issues/12)
(retention by segment).

Domain knowledge is as welcome as code. Several issues — segmented artifact
suites ([#13](https://github.com/orgoca/CompileML/issues/13)), informative
missingness ([#11](https://github.com/orgoca/CompileML/issues/11)) — need
someone who has governed a scorecard in production more than they need someone
who writes fast Python. Comment on the issue before opening a PR on those; the
design discussion is the work.

## Ground rules

1. **The spec is the contract.** Runtime, exporters, and validators implement
   [docs/ARTIFACT_SPEC.md](docs/ARTIFACT_SPEC.md). Behavior changes require a
   spec change in the same PR.
2. **`compileml.runtime` stays standard-library only.** A test parses every
   runtime module's imports; don't fight it.
3. **Claims are tests.** If a PR adds a guarantee to the docs, it adds the test
   that enforces it. If it can't be tested, it isn't claimed.
4. **No timestamps or randomness in hashed artifact content.** Identical
   inputs must produce identical hashes.

## Development setup

```bash
git clone https://github.com/orgoca/CompileML
cd compileml
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .[dev,xgboost,lightgbm]
pytest
```

The suite is deterministic and offline; `pytest` should pass everywhere.
The GnuCOBOL parity test runs automatically where `cobc` is installed and
skips elsewhere.

### Notebooks

CI executes the example notebooks on every push. A notebook may opt out by
setting `metadata.compileml.ci_execute = false` — reserved for rendered
galleries whose APIs are already covered by unit tests. If you change
`compileml.viz`, re-run `examples/04_visualization.ipynb` and commit the
regenerated figures.

## Style

- `ruff check` and `black --check` (line length 100) must pass.
- Match the surrounding code's docstring style (Google-ish, concise).
- Public API changes update `docs/reference/api.md` and the CHANGELOG.

## Reporting issues

For suspected parity or determinism bugs, include the artifact hash, the
`compileml inspect` output, and a minimal input row — those three usually
pin it down.
