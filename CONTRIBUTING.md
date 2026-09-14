# Contributing to CompileML

Thanks for your interest. CompileML aims for a small number of hard promises,
each enforced by tests — contributions are judged against that bar.

Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Where to start

The [open issues](https://github.com/orgoca/CompileML/issues) are written to be
picked up: each one states the problem, why it matters in a regulated decision
context, a proposed approach, acceptance criteria, and the files involved.

Two of them are, in my view, what stands between this being interesting
infrastructure and something a risk function could actually adopt:

| | |
|---|---|
| [#16](https://github.com/orgoca/CompileML/issues/16) | **Champion/challenger.** Replacing a model means answering who moves, by how much, and why. Both artifacts are hashed and exactly attributed, so the comparison can be exact and reproducible — the same decomposition that powers fairness and drift reports, applied across two artifacts instead of two populations. |
| [#39](https://github.com/orgoca/CompileML/issues/39) | **A scorecard base plus a residual whitebox.** On one large portfolio a plain WOE logistic regression outscored the compiled whitebox, and boosting on the logistic model's residual beat both. The representation already exists; what the issue needs before any spec change is evidence from a second dataset — which anyone with a real portfolio can supply. |

Smaller entry points: [#18](https://github.com/orgoca/CompileML/issues/18)
(WOE compatibility docs), [#31](https://github.com/orgoca/CompileML/issues/31)
(waterfall labels clipped at the canvas edge),
[#19](https://github.com/orgoca/CompileML/issues/19) (distillation cost in
expected-loss terms).

Domain knowledge is as welcome as code. Several issues — segmented artifact
suites ([#13](https://github.com/orgoca/CompileML/issues/13)), informative
missingness ([#11](https://github.com/orgoca/CompileML/issues/11)) — need
someone who has governed a scorecard in production more than they need someone
who writes fast Python. Comment before opening a PR on those; the
design discussion is the work.

## Questions and discussions

[Discussions](https://github.com/orgoca/CompileML/discussions) is for what is
not yet a piece of work someone can pick up and finish:

- **Q&A** — how do I…? The answer stays findable for the next person.
- **Ideas** — open design questions, such as whether CompileML should specify
  segmented artifact suites at all
  ([#13](https://github.com/orgoca/CompileML/issues/13)).
- **Show and tell** — what compiling cost on your data, or how you deployed it.

Issues are for bugs and scoped work. A discussion that settles into a concrete
change becomes an issue.

## Ground rules

1. **The spec is the contract.** Runtime, exporters, and validators implement
   [docs/ARTIFACT_SPEC.md](docs/ARTIFACT_SPEC.md). Behavior changes require a
   spec change in the same PR.
2. **Two sides, and only one of them is constrained.** Decide which side a
   change is on before writing it.
   - The **learning side** fits, compiles, bands and tunes models, and analyses
     decisions after they are made (`compileml.fairness`, monitoring). Any
     library is fine.
   - The **inference side** is where the artifact lives and makes the decision:
     `compileml.runtime`, and the code the exporters emit. It is
     standard-library only, always, and uses only simple arithmetic —
     integer addition, comparison and table lookup — so the same decision
     compiles to NumPy matrices, SQL and COBOL. A test parses every runtime
     module's imports; don't fight it.

   Code that makes or changes a decision is inference side. A spec change to
   scoring, banding, calibration or attribution has to stay expressible in
   that arithmetic, or it does not belong in the artifact.
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

Issues are open to anyone with a GitHub account — there is no approval step
and no bar to clear. Two templates exist to save you guessing what is
useful, and a blank issue stays available because a template should lower
the cost of reporting, not gate it.

For suspected parity or determinism bugs, include the artifact hash, the
`compileml inspect` output, and a minimal input row — those three usually
pin it down.

If you have put the library through its paces against your own data, the
**Findings** template is for that. Rough notes are welcome; a list of
observations reaches us, and a polished report that never gets written does
not. Link a document of your own if that is easier than filing separately —
[deburky/compileml-fraud-scoring](https://github.com/deburky/compileml-fraud-scoring)
is what that looks like in practice, and two bugs and several documentation
fixes came out of it.
