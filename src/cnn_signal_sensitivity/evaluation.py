"""Paired evaluation, saved per-image probabilities, and reproducible experiment records."""

from __future__ import annotations

import json
import platform
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from .models import WEIGHTS_NAME
from .transforms import validate_images


def seed_everything(seed=5550):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class Predictions:
    ids: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray
    image_mse: np.ndarray
    boundary_fraction: np.ndarray

    @property
    def predicted(self):
        return self.probabilities.argmax(axis=1)

    @property
    def correct(self):
        return self.predicted == self.targets

    def save(self, path):
        np.savez_compressed(
            path,
            ids=self.ids,
            targets=self.targets,
            probabilities=self.probabilities,
            image_mse=self.image_mse,
            boundary_fraction=self.boundary_fraction,
        )


def _check_pairs(clean, changed):
    if not np.array_equal(clean.ids, changed.ids) or not np.array_equal(
        clean.targets, changed.targets
    ):
        raise ValueError("Paired comparisons require identical image IDs and labels in order.")


@torch.inference_mode()
def predict(model, loader, condition, *, device=None, seed=5550):
    model.eval()
    device = device or next(model.parameters()).device
    collected = {
        name: [] for name in ["ids", "targets", "probabilities", "image_mse", "boundary_fraction"]
    }
    for images, targets, image_ids in loader:
        validate_images(images)
        changed = condition.apply(images, image_ids, seed=seed)
        if changed.shape != images.shape:
            raise ValueError(f"{condition.key} changed the input dimensions.")
        validate_images(changed)
        probabilities = model(changed.to(device)).softmax(dim=1).cpu()
        if probabilities.ndim != 2 or probabilities.shape != (len(images), 10):
            raise ValueError("The classifier must return ten logits per image.")
        if not torch.isfinite(probabilities).all():
            raise ValueError("Non-finite classifier output.")
        collected["ids"].append(image_ids.cpu().numpy())
        collected["targets"].append(targets.cpu().numpy())
        collected["probabilities"].append(probabilities.numpy())
        collected["image_mse"].append((changed - images).square().mean((1, 2, 3)).numpy())
        collected["boundary_fraction"].append(
            ((changed <= 0) | (changed >= 1)).float().mean((1, 2, 3)).numpy()
        )
    if not collected["ids"]:
        raise ValueError("Evaluation loader is empty.")
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    if len(np.unique(arrays["ids"])) != len(arrays["ids"]):
        raise ValueError("Repeated image IDs in evaluation loader.")
    order = np.argsort(arrays["ids"])
    return Predictions(**{key: value[order] for key, value in arrays.items()})


def summarize(clean, changed):
    _check_pairs(clean, changed)
    probabilities = changed.probabilities
    confidence = probabilities.max(axis=1)
    correct = changed.correct
    baseline_correct = clean.correct
    return {
        "n": len(correct),
        "accuracy_pct": 100 * correct.mean(),
        "drop_pp": 100 * (baseline_correct.mean() - correct.mean()),
        "flip_pct": 100 * (changed.predicted != clean.predicted).mean(),
        "retained_clean_correct_pct": 100 * correct[baseline_correct].mean()
        if baseline_correct.any()
        else np.nan,
        "confidence_mean": confidence.mean(),
        "confidence_wrong_mean": confidence[~correct].mean() if (~correct).any() else np.nan,
        "entropy_mean": -(probabilities * np.log(probabilities.clip(1e-12))).sum(axis=1).mean(),
        "image_mse_mean": changed.image_mse.mean(),
        "boundary_fraction_mean": changed.boundary_fraction.mean(),
    }


def per_class_accuracy(predictions, class_names):
    rows = []
    for label, name in enumerate(class_names):
        mask = predictions.targets == label
        rows.append(
            {
                "class": name,
                "n": int(mask.sum()),
                "accuracy_pct": 100 * predictions.correct[mask].mean() if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix(predictions, n_classes=10):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (predictions.targets, predictions.predicted), 1)
    return matrix


def paired_accuracy_interval(first, second, *, seed=5550, repetitions=2000):
    """Paired image bootstrap, second minus first accuracy, in percentage points."""
    _check_pairs(first, second)
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    differences = second.correct.astype(float) - first.correct.astype(float)
    generator = np.random.default_rng(seed)
    estimates = (
        np.array(
            [
                generator.choice(differences, size=len(differences), replace=True).mean()
                for _ in range(repetitions)
            ]
        )
        * 100
    )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "difference_pp": float(100 * differences.mean()),
        "lower_95_pp": float(lower),
        "upper_95_pp": float(upper),
    }


def run_suite(model, loader, conditions, output_dir, *, seed=5550, metadata=None, baseline=None):
    from .transforms import identity_condition

    conditions = list(conditions)
    if len({condition.key for condition in conditions}) != len(conditions):
        raise ValueError("Condition keys must be unique.")
    if not any(condition.key == "clean" for condition in conditions):
        conditions.insert(0, identity_condition())
    else:
        conditions.sort(key=lambda condition: condition.key != "clean")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions, rows = {}, []
    for index, condition in enumerate(conditions):
        result = (
            baseline
            if condition.key == "clean" and baseline is not None
            else predict(model, loader, condition, seed=seed)
        )
        predictions[condition.key] = result
        result.save(destination / f"{condition.key}.npz")
        row = {
            "condition": condition.key,
            "family": condition.family,
            "parameter": condition.parameter,
            "parameter_name": condition.parameter_name,
            "plot_kind": condition.plot_kind,
            **summarize(predictions["clean"], result),
        }
        rows.append(row)
        print(
            f"[{index + 1}/{len(conditions)}] {condition.key}: {row['accuracy_pct']:.2f}%",
            flush=True,
        )
    table = pd.DataFrame(rows)
    table.to_csv(destination / "summary.csv", index=False)
    record = {
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "weights": WEIGHTS_NAME,
        "decision_space": "Imagenette ten selected ImageNet logits; conditional ten-class softmax",
        "pipeline": "RGB -> resize shorter side 256 bilinear -> center crop 224 -> [0,1] "
        "-> transformation -> ImageNet normalization -> frozen ResNet18",
        "device": str(next(model.parameters()).device),
        "batch_size": loader.batch_size,
        "conditions": [
            {
                "key": c.key,
                "family": c.family,
                "parameter": c.parameter,
                "parameter_name": c.parameter_name,
                "stochastic": c.stochastic,
                "plot_kind": c.plot_kind,
                "settings": c.settings,
            }
            for c in conditions
        ],
        "notes": metadata or {},
    }
    (destination / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return table, predictions
