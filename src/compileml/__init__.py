"""CompileML — compile tree-ensemble models into deterministic decision artifacts.

The package splits into two halves with different dependency contracts:

- ``compileml.compile`` / ``compileml.artifact`` / ``compileml.bands`` /
  ``compileml.validate`` / ``compileml.export`` — the *compile side*, which may
  use numpy and scikit-learn.
- ``compileml.runtime`` — the *decision side*, which imports only the Python
  standard library. A compiled artifact can be scored, banded, calibrated, and
  explained with nothing but this subpackage (or a copy of it).
"""

from compileml.runtime import decide, load_artifact, verify_artifact

# The single source of truth for the package version. pyproject.toml reads
# this attribute at build time (setuptools dynamic version), so the wheel
# metadata, `compileml.__version__`, `compileml inspect`, and the
# `compileml_version` recorded inside every artifact can never disagree.
__version__ = "0.9.0"

__all__ = ["compile_selected", "decide", "load_artifact", "verify_artifact", "__version__"]


def __getattr__(name):
    # compile_selected is learning-side (NumPy, scikit-learn). Importing it
    # lazily keeps `import compileml` free of those dependencies, so the
    # runtime story - copy compileml/runtime anywhere - stays true of the root.
    if name == "compile_selected":
        from compileml.select import compile_selected

        return compile_selected
    raise AttributeError(f"module 'compileml' has no attribute {name!r}")
