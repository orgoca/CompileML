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
__version__ = "0.1.1"

__all__ = ["decide", "load_artifact", "verify_artifact", "__version__"]
