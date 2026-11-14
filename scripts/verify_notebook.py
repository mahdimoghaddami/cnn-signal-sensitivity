"""Execute the instructor notebook and save verified outputs into it, keeping a backup."""

import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "instructor/01_cnn_signal_sensitivity_solution.ipynb"
    notebook = nbformat.read(source, as_version=4)
    environment = dict(os.environ, MPLBACKEND="module://matplotlib_inline.backend_inline")
    client = NotebookClient(
        notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    # Use this interpreter directly, avoiding accidental selection of a system kernel.
    from jupyter_client import KernelManager

    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client.km = manager
    destination = ROOT / "outputs/lab/instructor_executed.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.execute(env=environment)
    finally:
        nbformat.write(notebook, destination)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "error"
    ]
    if errors:
        raise RuntimeError(f"Notebook contains errors: {errors}")
    backup_dir = ROOT / "outputs/notebook_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    shutil.copy2(source, backup_dir / f"{source.stem}_{stamp}.ipynb")
    nbformat.write(notebook, source)
    print(
        f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells: {destination}"
    )


if __name__ == "__main__":
    main()
