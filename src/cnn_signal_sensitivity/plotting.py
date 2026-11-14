"""Consistent figures with actual parameter values and paired visual examples."""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CLASS_NAMES

plt.rcParams.update(
    {"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
)


def image_grid(images, labels=None, *, columns=5):
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for i, (axis, image) in enumerate(zip(axes.flat, images)):
        axis.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1))
        if labels is not None:
            axis.set_title(CLASS_NAMES[int(labels[i])])
    fig.tight_layout()
    return fig


def show_input_output(original: torch.Tensor, transformed: torch.Tensor, output_title: str):
    """Display the first input/output pair at the same fixed RGB intensity scale."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for axis, batch, title in zip(
        axes, [original, transformed], ["Input image", output_title], strict=True
    ):
        axis.imshow(batch[0].detach().cpu().permute(1, 2, 0).clamp(0, 1))
        axis.set_title(title)
        axis.axis("off")
    return fig


def plot_curves(table, metric="accuracy_pct", *, families=None):
    families = (
        list(families)
        if families is not None
        else [family for family in table.family.unique() if family != "clean"]
    )
    if not families:
        raise ValueError("No transformation families to plot.")
    titles = {
        "accuracy_pct": "Accuracy (%)",
        "flip_pct": "Prediction flips (%)",
        "confidence_mean": "Mean ten-class confidence",
        "drop_pp": "Accuracy drop (pp)",
    }
    fig, axes = plt.subplots(
        math.ceil(len(families) / 3),
        3,
        figsize=(14, 3.7 * math.ceil(len(families) / 3)),
        squeeze=False,
    )
    clean = table.loc[table.condition == "clean", metric].iloc[0]
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, family in zip(axes.flat, families):
        axis.set_visible(True)
        subset = table[table.family == family]
        if subset.empty:
            raise ValueError(f"No results for family {family!r}.")
        categorical = "plot_kind" in subset and subset.plot_kind.eq("bar").all()
        if categorical:
            bars = axis.bar(subset.parameter.astype(str), subset[metric], color="#137c8b")
            for bar, value in zip(bars, subset[metric], strict=True):
                text = f"{value:.1f}%" if metric in {"accuracy_pct", "flip_pct"} else f"{value:.2f}"
                inside = metric in {"accuracy_pct", "flip_pct"} and value > 80
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value - 4 if inside else value + 1.5,
                    text,
                    ha="center",
                    va="top" if inside else "bottom",
                    color="white" if inside else "black",
                    fontsize=8,
                    fontweight="bold" if inside else "normal",
                )
            axis.tick_params(axis="x", rotation=45, labelsize=8)
            for label in axis.get_xticklabels():
                label.set_ha("right")
        else:
            numeric = subset.parameter.astype(float)
            axis.plot(numeric, subset[metric], "o-", color="#137c8b", linewidth=2)
            axis.set_xticks(numeric)
            if numeric.iloc[0] > numeric.iloc[-1]:
                axis.invert_xaxis()
        axis.axhline(clean, color="#9d4d27", linestyle="--", linewidth=1, label="clean")
        axis.set(
            title=family.replace("_", " "),
            xlabel=subset.parameter_name.iloc[0],
            ylabel=titles.get(metric, metric),
        )
        if metric in {"accuracy_pct", "flip_pct"}:
            axis.set_ylim(-2, 102)
        if metric == "confidence_mean":
            axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_class_changes(clean, results, keys, names=CLASS_NAMES):
    """Class-wise accuracy differences relative to the same clean images (percentage points)."""
    from .evaluation import _check_pairs

    differences = []
    for key in keys:
        changed = results[key]
        _check_pairs(clean, changed)
        row = []
        for label in range(len(names)):
            mask = clean.targets == label
            row.append(
                100 * (changed.correct[mask].mean() - clean.correct[mask].mean())
                if mask.any()
                else np.nan
            )
        differences.append(row)
    differences = np.asarray(differences)
    limit = max(5, float(np.nanmax(np.abs(differences))))
    fig, axis = plt.subplots(figsize=(12, max(3, 0.6 * len(keys) + 1.5)))
    shown = axis.imshow(differences, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set(xticks=range(len(names)), xticklabels=names, yticks=range(len(keys)), yticklabels=keys)
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
    for row in range(len(keys)):
        for column in range(len(names)):
            value = differences[row, column]
            axis.text(
                column,
                row,
                f"{value:+.0f}",
                ha="center",
                va="center",
                color="white" if abs(value) > limit * 0.55 else "black",
                fontsize=8,
            )
    axis.set_title("Accuracy change from clean, by class")
    fig.colorbar(shown, ax=axis, label="Changed minus clean accuracy (pp)")
    fig.tight_layout()
    return fig


def plot_confusion(matrix, names=CLASS_NAMES):
    fig, axis = plt.subplots(figsize=(9, 8))
    shown = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted class",
        ylabel="True class",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(len(names)):
        for col in range(len(names)):
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[row, col] > matrix.max() * 0.55 else "black",
            )
    fig.colorbar(shown, ax=axis, shrink=0.8, label="Image count")
    fig.tight_layout()
    return fig


@torch.inference_mode()
def transformation_gallery(model, images, labels, image_ids, conditions, *, seed=5550):
    fig, axes = plt.subplots(
        len(images),
        len(conditions),
        figsize=(3.2 * len(conditions), 3.8 * len(images)),
        squeeze=False,
        layout="constrained",
    )
    device = next(model.parameters()).device
    for column, condition in enumerate(conditions):
        changed = condition.apply(images, image_ids, seed=seed)
        probabilities = model(changed.to(device)).softmax(1).cpu()
        confidence, predicted = probabilities.max(1)
        for row in range(len(images)):
            axis = axes[row, column]
            axis.imshow(changed[row].cpu().permute(1, 2, 0).clamp(0, 1))
            axis.set_title(
                f"{condition.key}\nPred: {CLASS_NAMES[int(predicted[row])]} "
                f"({confidence[row]:.2f})\n"
                f"True: {CLASS_NAMES[int(labels[row])]} | ID {int(image_ids[row])}",
                fontsize=9,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    return fig
