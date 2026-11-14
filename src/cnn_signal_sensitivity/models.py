"""Frozen ten-logit ImageNet classifier with normalization in exactly one place."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .data import CLASS_NAMES, IMAGENET_INDICES

WEIGHTS_NAME = "ResNet18_Weights.IMAGENET1K_V1"


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RestrictedClassifier(nn.Module):
    """The decision and softmax are restricted to ten Imagenette categories."""

    def __init__(self, backbone, indices=IMAGENET_INDICES):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("class_indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        logits = self.backbone((images - self.mean) / self.std)
        return logits.index_select(1, self.class_indices)


def load_classifier(device=None, *, progress=True):
    weights = ResNet18_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]
    # Sanity-check names as well as integer indices against installed weight metadata.
    for index, name in zip(IMAGENET_INDICES, CLASS_NAMES, strict=True):
        if categories[index].lower() != name.lower():
            raise ValueError(f"Weight category mismatch: {index}: {categories[index]} != {name}")
    classifier = RestrictedClassifier(resnet18(weights=weights, progress=progress))
    return classifier.to(device or choose_device())
