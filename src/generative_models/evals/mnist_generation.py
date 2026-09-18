from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from generative_models.evals.mnist_classifier import (
    MNISTClassifier,
    freeze_mnist_classifier,
)


@dataclass(frozen=True)
class GeneratedMNISTMetrics:
    conditional_accuracy: float
    requested_class_confidence: float
    out_of_range_fraction: float
    confusion_matrix: Tensor


def load_frozen_mnist_classifier(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> MNISTClassifier:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model = MNISTClassifier(**checkpoint["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return freeze_mnist_classifier(model)


@torch.inference_mode()
def evaluate_generated_mnist(
    classifier: MNISTClassifier,
    samples: Tensor,
    requested_labels: Tensor,
) -> GeneratedMNISTMetrics:
    evaluation_images = samples.clamp(-1.0, 1.0)
    output = classifier(evaluation_images)
    probabilities = output.logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    requested_labels = requested_labels.to(device=samples.device)

    encoded = requested_labels * 10 + predictions
    confusion = torch.bincount(
        encoded.cpu(),
        minlength=100,
    ).reshape(10, 10)

    return GeneratedMNISTMetrics(
        conditional_accuracy=float((predictions == requested_labels).float().mean()),
        requested_class_confidence=float(
            probabilities.gather(1, requested_labels[:, None]).mean()
        ),
        out_of_range_fraction=float((samples.abs() > 1.0).float().mean()),
        confusion_matrix=confusion,
    )
