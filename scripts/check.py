"""Run everything CI gates on, in the order CI runs it.

The point is that there is one command to remember instead of five, and that
it fails where CI would fail rather than somewhere near it. If this passes,
the pull request goes green.

    python scripts/check.py             # the full gate
    python scripts/check.py --fix       # format and autofix first, then check
    python scripts/check.py --fast      # skip the docs build and notebooks

Cross-platform on purpose: contributors are not assumed to have ``make``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Mirrors .github/workflows/ci.yml. When that file changes, change this too —
# a local gate that has drifted from CI is worse than no local gate, because
# it is trusted.
LINT_PATHS = ["src", "tests", "benchmarks"]

CHECKS: list[tuple[str, list[str], bool]] = [
    # (label, argv after the interpreter, is_slow)
    ("ruff", ["-m", "ruff", "check", *LINT_PATHS], False),
    ("black", ["-m", "black", "--check", *LINT_PATHS], False),
    ("mypy", ["-m", "mypy", "src/compileml"], False),
    ("pytest", ["-m", "pytest", "-q"], False),
    ("docs", ["-m", "mkdocs", "build", "--strict"], True),
]

FIXES: list[tuple[str, list[str]]] = [
    ("black", ["-m", "black", *LINT_PATHS]),
    ("ruff --fix", ["-m", "ruff", "check", "--fix", *LINT_PATHS]),
]


def run(label: str, args: list[str]) -> tuple[str, bool, float, str]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return label, True, elapsed, ""
    if "No module named" in output:
        return label, False, elapsed, f"not installed — run: pip install -e .[dev,docs]\n{output}"
    return label, False, elapsed, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="format and autofix before checking")
    parser.add_argument("--fast", action="store_true", help="skip the slow docs build")
    args = parser.parse_args()

    if args.fix:
        for label, argv in FIXES:
            print(f"[fix] {label}")
            subprocess.run([sys.executable, *argv], cwd=ROOT)
        print()

    failures = []
    for label, argv, slow in CHECKS:
        if slow and args.fast:
            print(f"  skip  {label}")
            continue
        name, ok, elapsed, output = run(label, argv)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<8} {elapsed:5.1f}s")
        if not ok:
            failures.append((name, output))

    if args.fast:
        print("\n--fast skipped the docs build; CI does not.")

    if not failures:
        print("\nAll green. CI should agree.")
        return 0

    for name, output in failures:
        print(f"\n{'-' * 62}\n{name}\n{'-' * 62}")
        print(output.strip()[:4000])

    print(f"\n{len(failures)} check(s) failed.")
    if any(n in ("ruff", "black") for n, _ in failures):
        print("Formatting problems? `python scripts/check.py --fix` rewrites the files.")
    return 1


if __name__ == "__main__":
    if shutil.which(sys.executable) is None:  # pragma: no cover
        sys.exit("could not locate the running interpreter")
    raise SystemExit(main())
