from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml
import torch
from torch import device as TorchDevice
from argparse import ArgumentParser
import hashlib
import json

import matplotlib.pyplot as plt
import torchvision

from generative_models.data.mnist import (
    index_set_sha256,
    mnist_raw_file_sha256,
)
from generative_models.evals.mnist_classifier import freeze_mnist_classifier
from generative_models.training.mnist_classifier import (
    EpochRecord,
    MNISTClassifierTrainingConfig,
    train_mnist_classifier,
)
from generative_models.viz.points import save_figure, use_clean_style


@dataclass(frozen=True)
class TrainMNISTClassifierArgs:
    config: Path
    output_dir: Path
    device: TorchDevice


def load_config(path: Path) -> MNISTClassifierTrainingConfig:
    raw_config = yaml.safe_load(path.read_text())
    return MNISTClassifierTrainingConfig(**raw_config)


def _parse_device(v: str) -> TorchDevice:
    return torch.device(v)


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_history(history: list[EpochRecord], path: Path) -> Path:
    with path.open("w") as file:
        for record in history:
            file.write(json.dumps(asdict(record)) + "\n")
    return path


def save_training_curves(history: list[EpochRecord], path: Path) -> Path:
    use_clean_style()
    epochs = [record.epoch for record in history]
    training_loss = [record.training.loss for record in history]
    validation_loss = [record.validation.loss for record in history]
    training_accuracy = [record.training.accuracy for record in history]
    validation_accuracy = [record.validation.accuracy for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, training_loss, marker="o", label="training")
    axes[0].plot(epochs, validation_loss, marker="o", label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Classifier Loss")
    axes[0].legend()

    axes[1].plot(epochs, training_accuracy, marker="o", label="training")
    axes[1].plot(epochs, validation_accuracy, marker="o", label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.95, 1.0)
    axes[1].set_title("Classifier Accuracy")
    axes[1].legend()

    figure.tight_layout()
    output_path = save_figure(figure, path)
    plt.close(figure)
    return output_path


def save_confusion_matrix(confusion: torch.Tensor, path: Path) -> Path:
    use_clean_style()
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion.numpy(), cmap="Blues")
    axis.set_xlabel("predicted label")
    axis.set_ylabel("true label")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    axis.set_title("Official MNIST Test Confusion")
    figure.colorbar(image, ax=axis, label="examples")
    figure.tight_layout()
    output_path = save_figure(figure, path)
    plt.close(figure)
    return output_path


def parse_args() -> TrainMNISTClassifierArgs:
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mnist_classifier.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/mnist_classifier")
    )
    parser.add_argument("--device", type=_parse_device, default=torch.device("cpu"))

    ns = parser.parse_args()
    return TrainMNISTClassifierArgs(
        config=ns.config,
        output_dir=ns.output_dir,
        device=ns.device,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = train_mnist_classifier(config, device=args.device)
    freeze_mnist_classifier(result.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    config_path = args.output_dir / "config.yaml"
    history_path = args.output_dir / "history.jsonl"
    curves_path = args.output_dir / "training_curves.png"
    confusion_path = args.output_dir / "confusion_matrix.png"
    summary_path = args.output_dir / "summary.json"

    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    save_history(result.history, history_path)
    save_training_curves(result.history, curves_path)
    save_confusion_matrix(result.test.confusion_matrix, confusion_path)

    torch.save(
        {
            "model": {
                "feature_dim": config.feature_dim,
                "dropout": config.dropout,
            },
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in result.model.state_dict().items()
            },
            "selection": {
                "metric": "validation_accuracy",
                "best_epoch": result.best_epoch,
                "best_value": result.best_validation_accuracy,
            },
            "normalization": "uint8 [0,255] -> float32 [0,1] -> 2*x-1",
        },
        checkpoint_path,
    )
    checkpoint_digest = file_sha256(checkpoint_path)

    test_accuracy = result.test.metrics.accuracy
    summary = {
        "config": asdict(config),
        "parameter_count": parameter_count(result.model),
        "feature_dim": result.model.feature_dim,
        "selection": {
            "split": "validation",
            "metric": "accuracy",
            "best_epoch": result.best_epoch,
            "best_validation_accuracy": result.best_validation_accuracy,
        },
        "official_test": {
            "evaluation_policy": (
                "once per training run, after validation checkpoint selection"
            ),
            "loss": result.test.metrics.loss,
            "accuracy": test_accuracy,
            "confusion_matrix_rows": "true_label",
            "confusion_matrix_columns": "predicted_label",
            "confusion_matrix": result.test.confusion_matrix.tolist(),
        },
        "split_provenance": {
            "split_seed": config.split_seed,
            "training_index_sha256": index_set_sha256(result.split.train_indices),
            "validation_index_sha256": index_set_sha256(
                result.split.validation_indices
            ),
        },
        "raw_file_sha256": mnist_raw_file_sha256(config.data_root),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_digest,
        },
        "artifacts": {
            "config": str(config_path),
            "history": str(history_path),
            "training_curves": str(curves_path),
            "confusion_matrix": str(confusion_path),
        },
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "acceptance": {
            "minimum_test_accuracy": config.minimum_test_accuracy,
            "passed": test_accuracy >= config.minimum_test_accuracy,
        },
        "interpretation": {
            "role": "frozen evaluator for generated MNIST images",
            "not_used_for": [
                "classifier guidance",
                "classifier-free guidance",
                "generator checkpoint selection",
            ],
            "limitation": (
                "classifier confidence and features are not perceptual truth"
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {summary_path}")
    print(
        f"best_epoch={result.best_epoch} "
        f"validation_accuracy={result.best_validation_accuracy:.4f} "
        f"test_accuracy={test_accuracy:.4f}"
    )

    if test_accuracy < config.minimum_test_accuracy:
        raise RuntimeError(
            "classifier missed the declared test-accuracy acceptance gate; "
            f"expected >= {config.minimum_test_accuracy:.4f}, "
            f"got {test_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
