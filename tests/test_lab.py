"""Numerical and experimental-design checks that do not download data or weights."""

import json
from types import SimpleNamespace
from zipfile import ZipFile

import nbformat
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
from cnn_signal_sensitivity.experiments import (
    core_conditions,
    experiment_conditions,
    extension_conditions,
)
from cnn_signal_sensitivity.models import RestrictedClassifier
from cnn_signal_sensitivity.transforms import (
    additive_noise,
    desaturate,
    identity_condition,
    line_response,
    otsu_threshold,
    point_response,
    resample,
    sobel_edges,
    threshold,
)
from instructor.reference import (
    gamma_transform,
    gaussian_blur,
    gaussian_kernel1d,
    quantize,
    unsharp_mask,
)

torch.set_num_threads(2)


def test_public_exercise_checks():
    checks.check_quantize(quantize)
    checks.check_gamma(gamma_transform)
    checks.check_kernel(gaussian_kernel1d)
    checks.check_unsharp(unsharp_mask)


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        (quantize, {"bits": 0}),
        (gamma_transform, {"gamma": 0}),
        (gaussian_blur, {"sigma": -1}),
        (unsharp_mask, {"sigma": 1, "amount": -1}),
    ],
)
def test_invalid_parameters(operation, kwargs):
    with pytest.raises(ValueError):
        operation(torch.zeros(1, 3, 16, 16), **kwargs)


def test_gaussian_separability_and_channels():
    image = torch.zeros(1, 3, 17, 17)
    image[0, 0, 8, 8] = 1
    output = gaussian_blur(image, 1)
    kernel = gaussian_kernel1d(1)
    torch.testing.assert_close(output[0, 0, 5:12, 5:12], kernel[:, None] * kernel[None, :])
    assert output[:, 1:].count_nonzero() == 0
    torch.testing.assert_close(output.sum(), torch.tensor(1.0))


def test_color_intensity_and_input_immutability():
    image = torch.rand(2, 3, 17, 17)
    original = image.clone()
    changed = desaturate(image, 0.3)
    torch.testing.assert_close(changed.mean(1), image.mean(1))
    gray = desaturate(image, 0)
    torch.testing.assert_close(gray[:, 0], gray[:, 1])
    unsharp_mask(image, sigma=1, amount=2)
    torch.testing.assert_close(image, original)


def test_sampling_identity_and_aliasing_control():
    checkerboard = (torch.arange(32)[:, None] + torch.arange(32)[None, :]).remainder(2).float()
    image = checkerboard.expand(1, 3, 32, 32).clone()
    torch.testing.assert_close(resample(image, 32), image)
    assert not torch.allclose(
        resample(image, 11, antialias=True), resample(image, 11, antialias=False)
    )


def test_chapter10_detectors_and_thresholds():
    constant = torch.ones(1, 3, 17, 17) * 0.4
    for operator in [point_response, line_response, sobel_edges]:
        torch.testing.assert_close(
            operator(constant), torch.zeros_like(constant), atol=1e-6, rtol=0
        )
    impulse = torch.zeros_like(constant)
    impulse[:, :, 8, 8] = 1
    assert point_response(impulse)[0, 0, 8, 8] == 1
    horizontal = torch.zeros_like(constant)
    horizontal[:, :, 8, :] = 1
    assert line_response(horizontal, "horizontal")[0, 0, 8, 8] == 1
    assert line_response(horizontal, "vertical")[0, 0, 8, 8] == 0
    bimodal = torch.cat([torch.full((1, 3, 17, 8), 0.2), torch.full((1, 3, 17, 9), 0.8)], dim=3)
    expected = threshold(bimodal, 0.5)
    torch.testing.assert_close(otsu_threshold(bimodal), expected)
    assert otsu_threshold(constant).count_nonzero() == 0


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


def test_noise_is_independent_of_batching():
    condition = extension_conditions("noise")[3]
    images = torch.full((3, 3, 17, 17), 0.5)
    ids = torch.tensor([10, 20, 30])
    batched = condition.apply(images, ids, seed=7)
    individual = torch.cat(
        [
            condition.apply(image[None], image_id[None], seed=7)
            for image, image_id in zip(images, ids)
        ]
    )
    torch.testing.assert_close(batched, individual, atol=0, rtol=0)
    assert not torch.equal(batched, condition.apply(images, ids, seed=8))
    red = additive_noise(images, 0.1, torch.Generator().manual_seed(3), channel=0)
    torch.testing.assert_close(red[:, 1:], images[:, 1:])


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


def test_run_exports_and_all_supplied_conditions(tmp_path):
    model = RestrictedClassifier(ToyBackbone())
    table, results = run_suite(
        model, DataLoader(ToyDataset(), batch_size=3), [identity_condition()], tmp_path
    )
    assert len(table) == 1 and len(results["clean"].ids) == 10
    assert np.load(tmp_path / "clean.npz")["probabilities"].shape == (10, 10)
    assert json.loads((tmp_path / "run.json").read_text())["weights"].endswith("IMAGENET1K_V1")
    for name in ["chapter10", "geometry", "noise"]:
        for condition in extension_conditions(name):
            output = condition.apply(torch.rand(1, 3, 32, 32), torch.tensor([5]))
            assert output.shape == (1, 3, 32, 32) and output.min() >= 0 and output.max() <= 1
    core = core_conditions(quantize, gamma_transform, gaussian_blur, unsharp_mask)
    assert len(core) == 36
    assert len({condition.key for condition in core}) == len(core)


def test_notebook_scaffolding_and_solution_free_archive(tmp_path):
    from scripts.build_notebooks import ROOT, notebook
    from scripts.package_student import package_student

    starter, solution = notebook(False), notebook(True)
    for document in [starter, solution]:
        nbformat.validate(document)
        for cell in document.cells:
            if cell.cell_type == "code":
                compile(cell.source, "<notebook cell>", "exec")
    assert sum("exercise" in cell.metadata.get("tags", []) for cell in starter.cells) == 5
    assert sum("raise NotImplementedError" in cell.source for cell in starter.cells) == 5
    assert not any("raise NotImplementedError" in cell.source for cell in solution.cells)
    prose = [cell.source for cell in starter.cells if cell.cell_type == "markdown"]
    assert sum(len(text.split()) for text in prose) < 800
    assert max(len(text.split()) for text in prose) < 100
    assert not any("chapter" in text.lower() or "§" in text for text in prose)
    archive_path = package_student(tmp_path / "student.zip")
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "notebooks/01_cnn_signal_sensitivity.ipynb" in names
        assert not any(
            name.startswith(("instructor/", "data/", "tests/", "scripts/")) for name in names
        )
        assert (
            "INSTRUCTOR COPY"
            not in archive.read("notebooks/01_cnn_signal_sensitivity.ipynb").decode()
        )
    assert (ROOT / "uv.lock").is_file()


def test_unified_suite_and_categorical_settings():
    conditions = experiment_conditions(quantize, gamma_transform, gaussian_blur, unsharp_mask)
    assert len(conditions) == len({c.key for c in conditions}) == 54
    assert sum(c.key == "clean" for c in conditions) == 1
    assert {"grayscale", "point", "sobel", "otsu", "threshold_0.5"} <= {c.key for c in conditions}
    assert all(c.plot_kind == "bar" for c in conditions if c.family == "detectors")
    selected = experiment_conditions(
        quantize, gamma_transform, gaussian_blur, unsharp_mask, groups=["thresholding"]
    )
    assert {c.family for c in selected} == {"clean", "thresholding"}
    for groups in [[], ["sampling", "sampling"], ["unknown"]]:
        with pytest.raises(ValueError):
            experiment_conditions(
                quantize, gamma_transform, gaussian_blur, unsharp_mask, groups=groups
            )


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
    conditions = experiment_conditions(
        quantize, gamma_transform, gaussian_blur, unsharp_mask, groups=["thresholding", "gamma"]
    )
    table, _ = run_suite(model, DataLoader(ToyDataset(), batch_size=5), conditions, tmp_path)
    fig = plot_curves(table)
    categorical = next(axis for axis in fig.axes if axis.get_title() == "thresholding")
    numeric = next(axis for axis in fig.axes if axis.get_title() == "gamma brighten")
    assert len(categorical.patches) == 6
    assert len(numeric.patches) == 0 and len(numeric.lines) == 2
    plt.close(fig)


def test_four_lab_operators_and_checks():
    from cnn_signal_sensitivity.experiments import lab_conditions
    from instructor.reference import affine_rotation, histogram_equalize, sample_image

    checks.check_sampling(sample_image)
    checks.check_affine(affine_rotation)
    checks.check_histogram(histogram_equalize)
    conditions = lab_conditions(sample_image, quantize, affine_rotation, histogram_equalize)
    assert len(conditions) == len({c.key for c in conditions}) == 18
    assert {c.family for c in conditions} == {
        "clean",
        "sampling",
        "quantization",
        "affine_rotation",
        "histogram_equalization",
    }
    image = torch.rand(2, 3, 17, 19)
    original = image.clone()
    for condition in conditions:
        changed = condition.apply(image, torch.tensor([0, 1]))
        assert changed.shape == image.shape and changed.dtype == image.dtype
        assert changed.min() >= 0 and changed.max() <= 1
    torch.testing.assert_close(image, original)


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
    assert "lab_conditions(sample_image, quantize, affine_rotation, histogram_equalize)" in text
    assert "already trained on ImageNet" in text


def test_each_exercise_has_annotations_a_reminder_and_an_immediate_image_demo():
    import ast

    from scripts.build_notebooks import notebook

    for solution in [False, True]:
        document = notebook(solution)
        exercises = [
            (index, cell)
            for index, cell in enumerate(document.cells)
            if "exercise" in cell.metadata.get("tags", [])
        ]
        assert len(exercises) == 5
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
            if function.name != "image_histogram":
                assert "show_input_output" in calls


def test_image_histograms_and_independent_equalization(monkeypatch):
    from instructor import reference

    checks.check_image_histogram(reference.image_histogram)
    color = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
    counts = reference.image_histogram(color)
    assert counts.shape == (1, 256) and counts[0, 85] == 1 and counts.sum() == 1
    images = torch.rand(2, 3, 7, 9)
    original = images.clone()
    counts = reference.image_histogram(images)
    assert counts.dtype == torch.long and counts.device == images.device
    assert counts.sum(1).tolist() == [63, 63]
    torch.testing.assert_close(images, original)
    def unavailable_histogram(batch):
        raise AssertionError("Equalization must not call image_histogram.")

    monkeypatch.setattr(reference, "image_histogram", unavailable_histogram)
    checks.check_histogram(reference.histogram_equalize)
