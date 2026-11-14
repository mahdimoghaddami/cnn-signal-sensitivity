"""Create a student archive using an explicit allowlist; never include solutions/data."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def package_student(destination=None):
    destination = Path(destination or ROOT / "dist/cnn-signal-sensitivity-student.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [
        ROOT / name
        for name in [
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            ".gitignore",
            "ASSIGNMENT.md",
            "notebooks/01_cnn_signal_sensitivity.ipynb",
        ]
    ]
    files.extend(sorted((ROOT / "src/cnn_signal_sensitivity").glob("*.py")))
    if not all(file.is_file() for file in files):
        raise FileNotFoundError("Build notebooks and sync dependencies before packaging.")
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(ROOT))
        archive.write(ROOT / "docs/STUDENT_README.md", "README.md")
    print(destination)
    return destination


if __name__ == "__main__":
    package_student()
