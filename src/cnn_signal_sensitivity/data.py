"""Fixed model inputs and class-balanced, disjoint Imagenette subsets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

WNIDS = (
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
)
CLASS_NAMES = (
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
)
IMAGENET_INDICES = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)


def input_transform():
    """Match ResNet18 IMAGENET1K_V1 resize/crop; normalize only inside the model."""
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


class IndexedSubset(Dataset):
    """Return image, ten-class target, and stable index in the original val dataset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(map(int, indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        image, target = self.dataset[original]
        return image, target, original


class CachedInputs(Dataset):
    """Materialize fixed inputs once in CPU RAM; no repeated JPEG decoding per condition."""

    def __init__(self, dataset):
        samples = [dataset[index] for index in range(len(dataset))]
        if not samples:
            raise ValueError("Cannot cache an empty dataset.")
        self.images = torch.stack([sample[0] for sample in samples])
        self.targets = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
        self.ids = torch.tensor([sample[2] for sample in samples], dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # A clone protects the cached original even if a student mutates their input.
        return self.images[index].clone(), self.targets[index], self.ids[index]


def _sample_records(dataset):
    # TorchVision stores these records privately; fail clearly if its API changes.
    samples = getattr(dataset, "_samples", None)
    if samples is None:
        raise RuntimeError("Imagenette sample API changed. Use the supplied uv.lock.")
    image_root = Path(dataset._image_root)
    return [
        {"path": Path(filename).relative_to(image_root).as_posix(), "target": int(target)}
        for filename, target in samples
    ]


def make_manifest(dataset, *, dev_per_class=20, eval_per_class=100, seed=5550):
    if dev_per_class < 1 or eval_per_class < 1:
        raise ValueError("Both subsets must contain at least one image per class.")
    records = _sample_records(dataset)
    generator = torch.Generator().manual_seed(seed)
    development, evaluation = [], []
    for label in range(len(CLASS_NAMES)):
        indices = [i for i, row in enumerate(records) if row["target"] == label]
        needed = dev_per_class + eval_per_class
        if len(indices) < needed:
            raise ValueError(f"Class {label} has {len(indices)} images; requested {needed}.")
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in permutation]
        development.extend(shuffled[:dev_per_class])
        evaluation.extend(shuffled[dev_per_class:needed])
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {
        "schema_version": 1,
        "dataset": "imagenette2-320/val",
        "seed": seed,
        "dev_per_class": dev_per_class,
        "eval_per_class": eval_per_class,
        "sample_list_sha256": fingerprint,
        "wnids": list(WNIDS),
        "development": development,
        "evaluation": evaluation,
        "samples": records,
    }


def load_imagenette(
    root="data",
    *,
    download=False,
    dev_per_class=20,
    eval_per_class=100,
    seed=5550,
    manifest_path="data/subsets.json",
):
    dataset = datasets.Imagenette(
        root=root,
        split="val",
        size="320px",
        download=download,
        transform=input_transform(),
    )
    if tuple(dataset.wnids) != WNIDS:
        raise ValueError("Unexpected Imagenette class order; refusing an incorrect logit mapping.")
    manifest = make_manifest(
        dataset,
        dev_per_class=dev_per_class,
        eval_per_class=eval_per_class,
        seed=seed,
    )
    destination = Path(manifest_path)
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                f"{destination} describes different data or settings. "
                "Choose a new manifest_path to keep experiments reproducible."
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return (
        IndexedSubset(dataset, manifest["development"]),
        IndexedSubset(dataset, manifest["evaluation"]),
        manifest,
    )


def case_studies(subset):
    """One fixed example per class, selected without using model correctness."""
    chosen = {}
    for image, target, image_id in subset:
        target, image_id = int(target), int(image_id)
        if target not in chosen:
            chosen[target] = (image, target, image_id)
        if len(chosen) == len(CLASS_NAMES):
            break
    ordered = [chosen[k] for k in sorted(chosen)]
    return (
        torch.stack([row[0] for row in ordered]),
        torch.tensor([row[1] for row in ordered]),
        torch.tensor([row[2] for row in ordered]),
    )


def load_lab_data(root="data", *, download=True, seed=5550):
    """One fixed, balanced set of 200 images; no student-facing train/test modes."""
    root = Path(root)
    dataset = datasets.Imagenette(
        root=root, split="val", size="320px", download=download, transform=input_transform()
    )
    if tuple(dataset.wnids) != WNIDS:
        raise ValueError("Unexpected Imagenette class order.")
    records = _sample_records(dataset)
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in range(10):
        indices = [i for i, row in enumerate(records) if row["target"] == label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[i] for i in order[:20])
    manifest = {
        "dataset": "imagenette2-320/val",
        "seed": seed,
        "indices": selected,
        "samples": [records[i] for i in selected],
    }
    destination = root / "lab_subset.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != manifest:
            raise ValueError("The saved lab subset differs from the requested data.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return IndexedSubset(dataset, selected)
