# Releasing CompileML

Maintainer checklist. Publishing uses PyPI **trusted publishing** — no API
tokens stored anywhere.

## One-time setup

1. **GitHub**: push the repo to `github.com/orgoca/CompileML`; create two
   [environments](../../settings/environments) named `testpypi` and `pypi`
   (optionally add yourself as required reviewer on `pypi`).
2. **TestPyPI** (test.pypi.org → Account → Publishing): add a *pending*
   trusted publisher — project `compileml`, owner `orgoca`, repo `CompileML`,
   workflow `release.yml`, environment `testpypi`.
3. **PyPI** (pypi.org → Account → Publishing): same, with environment `pypi`.
4. **GitHub Pages**: Settings → Pages → deploy from branch `gh-pages`
   (created by the docs workflow on first push to main).

## TestPyPI dry run

1. Actions → `release` → *Run workflow* (target: `testpypi`).
2. Verify: `pip install -i https://test.pypi.org/simple/ --extra-index-url
   https://pypi.org/simple compileml` in a scratch venv, then
   `compileml --help` and a quick `import compileml`.

## Releasing vX.Y.Z

1. Bump `__version__` in `src/compileml/__init__.py` — the single source of
   truth; pyproject reads it at build time — and move the `[Unreleased]`
   CHANGELOG section under the new version with the date. Update `version:`
   and `date-released:` in `CITATION.cff` to match; Zenodo reads that file
   when it archives the release, so a stale version there is published.
   (`__version__` stays the single source of truth for the *package* — this
   is citation metadata, which Zenodo and GitHub read separately.)
2. Commit: `release: vX.Y.Z`, then tag and push:
   ```bash
   git tag vX.Y.Z
   git push && git push --tags
   ```
   If the bump went through a pull request, tag the **merge commit** — not
   whatever `main` pointed at when the PR was opened. Verify before
   publishing, because the mismatch is otherwise invisible until the upload
   fails:
   ```bash
   git show vX.Y.Z:src/compileml/__init__.py | grep __version__
   ```
3. Create a GitHub Release from the tag (paste the CHANGELOG section).
   Publishing to PyPI triggers automatically from the release event.
4. Post-release: bump `__version__` in `src/compileml/__init__.py` to the
   next `.dev0` and commit.

## Release gates (all enforced by CI before you ever tag)

- full matrix green (3 OS × py3.10–3.13), including the committed
  determinism oracle and the GnuCOBOL run-parity test
- cross-OS build-determinism job: identical artifact hashes on all three OSes
- lint, docs `--strict`, notebook execution, `twine check`
