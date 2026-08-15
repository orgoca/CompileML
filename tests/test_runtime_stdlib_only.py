"""Enforce the runtime's headline property: standard library only.

'No ML runtime' is a checkable fact, not a marketing line. This test
parses every module under compileml/runtime and fails if any import
resolves outside the Python standard library or the runtime itself.
"""

import ast
import sys
from pathlib import Path

import compileml.runtime as rt

ALLOWED_PREFIX = "compileml.runtime"


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module)
    return roots


def test_runtime_imports_stdlib_only():
    pkg_dir = Path(rt.__file__).parent
    offenders: list[tuple[str, str]] = []
    for module_path in sorted(pkg_dir.glob("*.py")):
        for name in _imported_roots(module_path):
            if name == "compileml.runtime" or name.startswith(ALLOWED_PREFIX + "."):
                continue
            root = name.split(".")[0]
            if root not in sys.stdlib_module_names:
                offenders.append((module_path.name, name))
    assert not offenders, f"non-stdlib imports in runtime: {offenders}"


def test_runtime_objects_not_backed_by_numpy_or_sklearn():
    """Nothing exported by runtime modules may originate in numpy/sklearn."""
    runtime_modules = [
        m
        for m in list(sys.modules.values())
        if getattr(m, "__name__", "").startswith("compileml.runtime")
    ]
    assert runtime_modules, "runtime modules should be imported by this test session"
    for mod in runtime_modules:
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name, None)
            mod_name = getattr(attr, "__module__", "") or ""
            assert not mod_name.startswith(
                ("numpy", "sklearn")
            ), f"{mod.__name__}.{attr_name} comes from {mod_name}"
