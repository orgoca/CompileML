# Contributing to CompileML

Thanks for your interest. CompileML aims for a small number of hard promises,
each enforced by tests — contributions are judged against that bar.

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

## Style

- `ruff check` and `black --check` (line length 100) must pass.
- Match the surrounding code's docstring style (Google-ish, concise).
- Public API changes update `docs/reference/api.md` and the CHANGELOG.

## Reporting issues

For suspected parity or determinism bugs, include the artifact hash, the
`compileml inspect` output, and a minimal input row — those three usually
pin it down.
