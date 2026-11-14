"""Dataset/weight preparation without running or solving the student exercises."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str((args.root / "models").resolve())
    from .data import load_lab_data
    from .models import load_classifier

    dataset = load_lab_data(args.root)
    model = load_classifier(device="cpu")
    print(
        f"Ready: {len(dataset)} lab images; "
        f"{type(model.backbone).__name__} weights cached under {args.root / 'models'}."
    )


if __name__ == "__main__":
    main()
