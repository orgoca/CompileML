"""Execute every example notebook; fail CI on any cell error.

A notebook opts out by setting ``metadata.compileml.ci_execute = false`` in
its own notebook metadata — the declaration travels with the file, so it
survives renames and is visible to anyone reading the notebook. Opted-out
notebooks still ship committed outputs; the APIs they demonstrate are
covered by the unit tests.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

failed = False
for path in sorted(EXAMPLES.glob("*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    if nb.metadata.get("compileml", {}).get("ci_execute") is False:
        print(f"skipping {path.name} (ci_execute: false)")
        continue
    print(f"executing {path.name} …", flush=True)
    try:
        NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(EXAMPLES)}}
        ).execute()
        print(f"  OK ({sum(1 for c in nb.cells if c.cell_type == 'code')} code cells)")
    except Exception as exc:  # noqa: BLE001 — report and continue to next notebook
        print(f"  FAILED: {exc}")
        failed = True

sys.exit(1 if failed else 0)
