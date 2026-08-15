"""Execute every example notebook; fail CI on any cell error."""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

failed = False
for path in sorted(EXAMPLES.glob("*.ipynb")):
    print(f"executing {path.name} …", flush=True)
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(EXAMPLES)}}
        ).execute()
        print(f"  OK ({sum(1 for c in nb.cells if c.cell_type == 'code')} code cells)")
    except Exception as exc:  # noqa: BLE001 — report and continue to next notebook
        print(f"  FAILED: {exc}")
        failed = True

sys.exit(1 if failed else 0)
