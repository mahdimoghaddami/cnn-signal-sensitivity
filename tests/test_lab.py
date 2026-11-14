"""Numerical and experimental-design checks that do not download data or weights."""

import ast
import json
from types import SimpleNamespace
from zipfile import ZipFile

import nbformat
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import gaussian_blur as library_gaussian

from cnn_signal_sensitivity import checks
from cnn_signal_sensitivity.data import (
    CLASS_NAMES,
    IMAGENET_INDICES,
    WNIDS,
    CachedInputs,
    make_manifest,
)
from cnn_signal_sensitivity.evaluation import (
    Predictions,
    paired_accuracy_interval,
    predict,
    run_suite,
    summarize,
)
from cnn_signal_sensitivity.experiments import lab_conditions
from cnn_signal_sensitivity.models import RestrictedClassifier
from cnn_signal_sensitivity.transforms import grayscale, identity_condition
from instructor.reference import (
    affine_rotation,
    gaussian_smooth,
    histogram_equalize,
    image_histogram,
    laplacian_sharpen,
    quantize,
    sample_image,
)

torch.set_num_threads(2)


def test_public_exercise_checks_pass_for_reference():
    checks.check_sampling(sample_image)
    checks.check_quantize(quantize)
    checks.check_affine(affine_rotation)
    checks.check_image_histogram(image_histogram)
    checks.check_histogram(histogram_equalize)
    checks.check_gaussian_smooth(gaussian_smooth)
    checks.check_laplacian_sharpen(laplacian_sharpen)


def test_public_checks_catch_common_bugs():
    # Batch-mixing histogram equalization bug must fail check_histogram.
    def batch_mixing_equalize(images):
        gray = images.mean(dim=1, keepdim=True)
        bins = torch.round(gray * 255).long()
        hist = torch.bincount(bins.flatten(), minlength=256).to(images.dtype)
        cdf = hist.cumsum(0)
        minimum = cdf[hist > 0][0]
        mapping = ((cdf - minimum) / (cdf[-1] - minimum)).clamp(0, 1)
        return mapping[bins].expand(-1, 3, -1, -1).contiguous()

    with pytest.raises(AssertionError):
        checks.check_histogram(batch_mixing_equalize)

    # Zero-border rotation (missing 0.5 gray fill) must fail check_affine.
    def black_border_rotation(images, angle):
        if angle == 0:
            return images.clone()
        import math

        import torch.nn.functional as F

        batch, _, height, width = images.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=images.device, dtype=images.dtype),
            torch.arange(width, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        x, y = x - (width - 1) / 2, y - (height - 1) / 2
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        sx = cosine * x - sine * y + (width - 1) / 2
        sy = sine * x + cosine * y + (height - 1) / 2
        grid = torch.stack((2 * (sx + 0.5) / width - 1, 2 * (sy + 0.5) / height - 1), dim=-1)
        return F.grid_sample(
            images,
            grid.unsqueeze(0).expand(batch, -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).clamp(0, 1)

    with pytest.raises(AssertionError):
        checks.check_affine(black_border_rotation)

    # In-place input mutation must fail check_sampling.
    def mutating_sample(images, factor):
        out = sample_image(images, factor)
        images.zero_()
        return out

    with pytest.raises(AssertionError):
        checks.check_sampling(mutating_sample)


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        (sample_image, {"factor": 0}),
        (sample_image, {"factor": 1.5}),
        (quantize, {"bits": 0}),
        (quantize, {"bits": 9}),
        (quantize, {"bits": 1.5}),
        (affine_rotation, {"angle": float("nan")}),
        (affine_rotation, {"angle": float("inf")}),
        (gaussian_smooth, {"sigma": -1}),
        (gaussian_smooth, {"sigma": float("nan")}),
        (gaussian_smooth, {"sigma": float("inf")}),
        (gaussian_smooth, {"sigma": 6}),
        (laplacian_sharpen, {"amount": -1}),
        (laplacian_sharpen, {"amount": float("nan")}),
        (laplacian_sharpen, {"amount": float("inf")}),
    ],
)
def test_invalid_parameters(operation, kwargs):
    with pytest.raises(ValueError):
        operation(torch.zeros(1, 3, 16, 16), **kwargs)


def test_image_histograms_and_equalization_behavior():
    color = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
    counts = image_histogram(color)
    assert counts.shape == (1, 256) and counts[0, 85] == 1 and counts.sum() == 1
    images = torch.rand(2, 3, 7, 9)
    original = images.clone()
    counts = image_histogram(images)
    assert counts.dtype == torch.long and counts.device == images.device
    assert counts.sum(1).tolist() == [63, 63]
    equalized = histogram_equalize(images)
    assert equalized.shape == images.shape and equalized.dtype == images.dtype
    torch.testing.assert_close(equalized[:, 0], equalized[:, 1])
    torch.testing.assert_close(images, original)


def test_filter_exercises_and_full_lab_suite():
    image = torch.rand(2, 3, 17, 19, dtype=torch.float64)
    original = image.clone()
    torch.testing.assert_close(gaussian_smooth(image, 1), library_gaussian(image, [7, 7], [1, 1]))
    for output in [gaussian_smooth(image, 1), laplacian_sharpen(image, 0.5)]:
        assert output.shape == image.shape and output.dtype == image.dtype
        assert output.device == image.device and output.min() >= 0 and output.max() <= 1
    torch.testing.assert_close(image, original)

    conditions = lab_conditions(
        sample_image,
        quantize,
        affine_rotation,
        histogram_equalize,
        gaussian_smooth,
        laplacian_sharpen,
    )
    assert len(conditions) == len({c.key for c in conditions}) == 31
    assert {"quantization_1", "gaussian_4", "laplacian_2", "rotation_-45", "rotation_45"} <= {
        c.key for c in conditions
    }
    assert {c.family for c in conditions} == {
        "clean",
        "sampling",
        "quantization",
        "affine_rotation",
        "histogram_equalization",
        "gaussian_smoothing",
        "laplacian_sharpening",
    }
    image32 = image.float()
    original32 = image32.clone()
    for condition in conditions:
        output = condition.apply(image32, torch.tensor([0, 1]))
        assert output.shape == image32.shape and output.dtype == image32.dtype
        assert torch.isfinite(output).all() and output.min() >= 0 and output.max() <= 1
    torch.testing.assert_close(image32, original32)


def test_grayscale_control_matches_rgb_mean():
    image = torch.rand(2, 3, 8, 8)
    gray = grayscale(image)
    torch.testing.assert_close(gray[:, 0], image.mean(dim=1))
    torch.testing.assert_close(gray[:, 0], gray[:, 2])


def test_manifest_is_balanced_disjoint_and_stable(tmp_path):
    records = [
        (str(tmp_path / "val" / wnid / f"{i}.JPEG"), label)
        for label, wnid in enumerate(WNIDS)
        for i in range(8)
    ]
    fake = SimpleNamespace(_samples=records, _image_root=str(tmp_path / "val"))
    manifest = make_manifest(fake, dev_per_class=2, eval_per_class=3)
    assert manifest == make_manifest(fake, dev_per_class=2, eval_per_class=3)
    assert set(manifest["development"]).isdisjoint(manifest["evaluation"])
    for name, count in [("development", 2), ("evaluation", 3)]:
        labels = [records[index][1] for index in manifest[name]]
        assert all(labels.count(label) == count for label in range(10))
    with pytest.raises(ValueError):
        make_manifest(fake, dev_per_class=5, eval_per_class=5)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, images):
        return images.mean((1, 2, 3))[:, None] + torch.arange(1000, device=images.device)[None, :]


class ToyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.full((3, 17, 17), index / 10), index, 100 + index


def test_model_normalization_and_ten_logit_mapping():
    model = RestrictedClassifier(ToyBackbone())
    image = torch.full((2, 3, 17, 17), 0.5)
    expected = model.backbone((image - model.mean) / model.std)[:, list(IMAGENET_INDICES)]
    torch.testing.assert_close(model(image), expected)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert len(CLASS_NAMES) == model(image).shape[1] == 10


def _predictions(classes):
    probabilities = np.eye(10)[classes] * 0.9 + 0.01
    return Predictions(
        np.arange(len(classes)),
        np.array([0, 1, 2, 3]),
        probabilities,
        np.zeros(len(classes)),
        np.zeros(len(classes)),
    )


def test_metrics_distinguish_correcting_flips_from_introduced_errors():
    clean = _predictions([0, 1, 0, 0])
    changed = _predictions([0, 0, 2, 3])
    metrics = summarize(clean, changed)
    assert metrics["accuracy_pct"] == 75
    assert metrics["drop_pp"] == -25
    assert metrics["flip_pct"] == 75
    assert metrics["retained_clean_correct_pct"] == 50
    same = paired_accuracy_interval(clean, clean, repetitions=50)
    assert same == {"difference_pp": 0, "lower_95_pp": 0, "upper_95_pp": 0}
    changed.ids = changed.ids[::-1]
    with pytest.raises(ValueError):
        summarize(clean, changed)


def test_run_exports_and_lab_conditions(tmp_path):
    model = RestrictedClassifier(ToyBackbone())
    table, results = run_suite(
        model, DataLoader(ToyDataset(), batch_size=3), [identity_condition()], tmp_path
    )
    assert len(table) == 1 and len(results["clean"].ids) == 10
    assert np.load(tmp_path / "clean.npz")["probabilities"].shape == (10, 10)
    assert json.loads((tmp_path / "run.json").read_text())["weights"].endswith("IMAGENET1K_V1")


def test_notebook_scaffolding_and_solution_free_archive(tmp_path, monkeypatch):
    from scripts import package_student as pkg
    from scripts.build_notebooks import ROOT, notebook

    starter, solution = notebook(False), notebook(True)
    for document in [starter, solution]:
        nbformat.validate(document)
        for cell in document.cells:
            if cell.cell_type == "code":
                compile(cell.source, "<notebook cell>", "exec")
    assert sum("exercise" in cell.metadata.get("tags", []) for cell in starter.cells) == 7
    assert sum("raise NotImplementedError" in cell.source for cell in starter.cells) == 7
    assert not any("raise NotImplementedError" in cell.source for cell in solution.cells)
    prose = [cell.source for cell in starter.cells if cell.cell_type == "markdown"]
    assert sum(len(text.split()) for text in prose) < 800
    assert max(len(text.split()) for text in prose) < 100
    assert not any("chapter" in text.lower() or "§" in text for text in prose)

    # Package against the real repo into tmp_path (never writing to dist/).
    archive_path = pkg.package_student(tmp_path / "student.zip")
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "notebooks/visual_computing_lab.ipynb" in names
        assert "README.md" in names
        assert "src/cnn_signal_sensitivity/checks.py" in names
        assert names.count("assignment.pdf") == 1
        assert not any(
            name.startswith(("instructor/", "data/", "tests/", "scripts/")) for name in names
        )
        assert (
            "Instructor copy: includes completed implementations."
            not in archive.read("notebooks/visual_computing_lab.ipynb").decode()
        )
    assert (ROOT / "uv.lock").is_file()

    # Verify PDF instruction inclusion when a PDF is present under docs/.
    fake_root = tmp_path / "repo"
    (fake_root / "src/cnn_signal_sensitivity").mkdir(parents=True)
    (fake_root / "notebooks").mkdir()
    (fake_root / "docs/latex").mkdir(parents=True)
    for rel in ["pyproject.toml", "uv.lock", ".python-version", ".gitignore"]:
        (fake_root / rel).write_text("x", encoding="utf-8")
    (fake_root / "notebooks/visual_computing_lab.ipynb").write_text("{}", encoding="utf-8")
    (fake_root / "src/cnn_signal_sensitivity/__init__.py").write_text("", encoding="utf-8")
    (fake_root / "docs/STUDENT_README.md").write_text("readme", encoding="utf-8")
    (fake_root / "docs/latex/lab_instructions.pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(pkg, "ROOT", fake_root)
    pdf_archive = pkg.package_student(tmp_path / "student_with_pdf.zip")
    with ZipFile(pdf_archive) as archive:
        names = archive.namelist()
        assert "lab_instructions.pdf" in names
        assert "ASSIGNMENT.md" not in names


def test_cached_inputs_keep_originals_and_sample_identity():
    cached = CachedInputs(ToyDataset())
    assert len(cached) == 10
    image, target, image_id = cached[4]
    assert (int(target), int(image_id)) == (4, 104)
    image.zero_()
    torch.testing.assert_close(cached[4][0], ToyDataset()[4][0])


def test_reusing_baseline_avoids_extra_inference(tmp_path):
    class CountingBackbone(ToyBackbone):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            return super().forward(images)

    model = RestrictedClassifier(CountingBackbone())
    loader = DataLoader(ToyDataset(), batch_size=5)
    baseline = predict(model, loader, identity_condition())
    assert model.backbone.calls == 2
    _, results = run_suite(model, loader, [identity_condition()], tmp_path, baseline=baseline)
    assert model.backbone.calls == 2
    assert results["clean"] is baseline


def test_rebuild_archives_the_executed_notebook(tmp_path):
    from scripts.build_notebooks import notebook, write_notebook

    destination = tmp_path / "lab.ipynb"
    original = notebook(True)
    original.cells[3].outputs = [nbformat.v4.new_output("stream", name="stdout", text="My run\n")]
    nbformat.write(original, destination)
    original_bytes = destination.read_bytes()
    write_notebook(destination, notebook(True), backup_dir=tmp_path / "backups")
    backups = list((tmp_path / "backups").glob("*.ipynb"))
    assert len(backups) == 1 and backups[0].read_bytes() == original_bytes
    assert destination.read_bytes() != original_bytes


def test_numeric_and_categorical_plots_use_different_encodings(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cnn_signal_sensitivity.plotting import plot_curves

    model = RestrictedClassifier(ToyBackbone())
    conditions = lab_conditions(
        sample_image,
        quantize,
        affine_rotation,
        histogram_equalize,
        gaussian_smooth,
        laplacian_sharpen,
    )
    table, _ = run_suite(model, DataLoader(ToyDataset(), batch_size=5), conditions, tmp_path)
    fig = plot_curves(table)
    categorical = next(axis for axis in fig.axes if axis.get_title() == "histogram equalization")
    numeric = next(axis for axis in fig.axes if axis.get_title() == "sampling")
    assert len(categorical.patches) == 2 and len(categorical.texts) == 2
    assert len(numeric.patches) == 0 and len(numeric.lines) == 2
    plt.close(fig)


def test_notebook_only_evaluates_the_exercise_functions():
    from scripts.build_notebooks import notebook

    document = notebook(True)
    text = "\n".join(c.source for c in document.cells)
    for removed in [
        "MODE",
        "RUN_LABEL",
        "DEEP_DIVE",
        "Otsu",
        "Unsharp",
        "early_features",
        "## 5.",
        "## 6.",
    ]:
        assert removed not in text
    assert "gaussian_smooth," in text and "laplacian_sharpen," in text
    assert "already trained on ImageNet" in text
    assert "quantization_1" in text
    assert "gaussian_4" in text and "laplacian_2" in text
    assert "confidence_mean" in text and "image_mse_mean" in text


def test_each_exercise_has_annotations_a_reminder_and_an_immediate_image_demo():
    from scripts.build_notebooks import notebook

    for solution in [False, True]:
        document = notebook(solution)
        exercises = [
            (index, cell)
            for index, cell in enumerate(document.cells)
            if "exercise" in cell.metadata.get("tags", [])
        ]
        assert len(exercises) == 7
        for index, cell in exercises:
            function = ast.parse(cell.source).body[0]
            assert all(argument.annotation is not None for argument in function.args.args)
            assert function.returns is not None
            assert ast.get_docstring(function) and "TODO" not in ast.get_docstring(function)
            assert "A single image must keep its batch dimension" not in ast.get_docstring(function)
            demo = document.cells[index + 1]
            assert "demonstration" in demo.metadata.get("tags", [])
            calls = [
                node.func.id
                for node in ast.walk(ast.parse(demo.source))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            ]
            assert function.name in calls
            assert "imshow" in demo.source


def test_expanded_instructor_demonstrations():
    from scripts.build_notebooks import notebook

    demos = {
        cell.metadata["tags"][1]: cell.source
        for cell in notebook(True).cells
        if "demonstration" in cell.metadata.get("tags", [])
    }
    assert "factors = [2, 4, 8, 16]" in demos["sample_image"]
    assert "bit_depths = [6, 4, 2, 1]" in demos["quantize"]
    assert "angles = [-45, -20, 20, 45]" in demos["affine_rotation"]
    assert "sigmas = [0.5, 1, 2, 4]" in demos["gaussian_smooth"]
    assert "amounts = [0.25, 0.5, 1, 2]" in demos["laplacian_sharpen"]
    assert "grayscale_input" in demos["image_histogram"]
    assert "histogram_equalize(demo_input)" in demos["histogram_equalize"]
    assert "twice_equalized" not in demos["histogram_equalize"]
    comparison = next(
        cell.source
        for cell in notebook(True).cells
        if "histogram-comparison" in cell.metadata.get("tags", [])
    )
    assert "image_histogram(demo_output)" in comparison
    assert "twice_equalized" not in comparison
