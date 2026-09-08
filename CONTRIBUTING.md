# Contributing to CompileML

Thanks for your interest. CompileML aims for a small number of hard promises,
each enforced by tests — contributions are judged against that bar.

## Where to start

The [open issues](https://github.com/orgoca/CompileML/issues) are written to be
picked up: each one states the problem, why it matters in a regulated lending
context, a proposed approach, acceptance criteria, and the files involved.

Two of them are, in my view, what stands between this being interesting
infrastructure and something a risk function could actually adopt:

| | |
|---|---|
| [#9](https://github.com/orgoca/CompileML/issues/9) | **Stability monitoring.** Not another PSI implementation — the issue is explicit about what *not* to rebuild. The open work is decomposing score drift across features exactly, which the reconciliation identity makes possible and external tooling can only approximate. |
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

## Before you open a pull request

One command runs everything CI gates on, in the order CI runs it:

```bash
python scripts/check.py          # ruff, black, mypy, pytest, docs build
python scripts/check.py --fix    # format and autofix first, then check
python scripts/check.py --fast   # skip the slow docs build
```

If that passes, the pull request goes green. It exists because the gate is
five commands over three directories — `ruff check src tests benchmarks`,
`black --check src tests benchmarks`, `mypy src/compileml`, `pytest`,
`mkdocs build --strict` — and remembering four of five is the normal
outcome. Run the script instead.

`--fix` rewrites the files for you; you are not expected to match the
formatting by hand. Line length is 100.

### Or never think about it again

```bash
pip install pre-commit && pre-commit install
```

`black` and `ruff` then run on every commit and fix the files before they
are committed, so formatting cannot be the reason a pull request goes red.
Optional, and the most useful five seconds you will spend on this repo.

## Style

- Match the surrounding code's docstring style (Google-ish, concise).
- Public API changes update `docs/reference/api.md` and the CHANGELOG.
- **A test should fail if the claim it makes stops being true.** For a bug
  fix, that usually means reproducing the failure the way a user met it
  rather than unit-testing the function you changed. A SQL generation bug
  wants a test that *executes* the SQL; an attribution bug wants one that
  checks the reconciliation identity. The existing tests are the reference —
  `tests/test_export.py` executes generated SQL against SQLite and compiles
  the generated COBOL where `cobc` is available.

## Reporting issues

For suspected parity or determinism bugs, include the artifact hash, the
`compileml inspect` output, and a minimal input row — those three usually
pin it down.
