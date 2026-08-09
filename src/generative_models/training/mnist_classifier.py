from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from generative_models.data.mnist import (
    MNIST_NUM_CLASSES,
    MNISTEvaluatorSplit,
    load_mnist_split,
    make_mnist_evaluator_split,
)
from generative_models.evals.mnist_classifier import MNISTClassifier
from generative_models.seeds import seed_everything


@dataclass(frozen=True)
class MNISTClassifierTrainingConfig:
    data_root: str = "data/mnist"
    download: bool = True
    split_seed: int = 2026
    validation_size: int = 5000
    training_seed: int = 42
    batch_size: int = 128
    evaluation_batch_size: int = 512
    num_workers: int = 0
    num_epochs: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    feature_dim: int = 128
    dropout: float = 0.2
    minimum_test_accuracy: float = 0.99


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class EvaluationResult:
    metrics: ClassificationMetrics
    confusion_matrix: Tensor


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    training: ClassificationMetrics
    validation: ClassificationMetrics


@dataclass(frozen=True)
class MNISTClassifierLoaders:
    training: DataLoader
    validation: DataLoader
    test: DataLoader
    split: MNISTEvaluatorSplit


@dataclass(frozen=True)
class MNISTClassifierTrainingResult:
    model: MNISTClassifier
    history: list[EpochRecord]
    best_epoch: int
    best_validation_accuracy: float
    test: EvaluationResult
    split: MNISTEvaluatorSplit


def confusion_matrix_from_predictions(
    predictions: Tensor,
    targets: Tensor,
    *,
    num_classes: int = MNIST_NUM_CLASSES,
) -> Tensor:
    """
    generate the confusion matrix based on prediction label and the target label
    """
    encoded_pairs = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    return torch.bincount(
        encoded_pairs.cpu(),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def build_mnist_classifier_loaders(
    config: MNISTClassifierTrainingConfig,
    *,
    device: torch.device,
) -> MNISTClassifierLoaders:
    training_dataset = load_mnist_split(
        config.data_root, train=True, download=config.download
    )
    test_dataset = load_mnist_split(
        config.data_root, train=False, download=config.download
    )
    split = make_mnist_evaluator_split(
        num_examples=len(training_dataset),
        validation_size=config.validation_size,
        seed=config.split_seed,
    )
    training_subset = Subset(training_dataset, split.train_indices.tolist())
    validation_subset = Subset(training_dataset, split.validation_indices.tolist())
    shuffle_generator = torch.Generator().manual_seed(config.training_seed)
    loader_options = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    return MNISTClassifierLoaders(
        training=DataLoader(
            dataset=training_subset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=shuffle_generator,
            **loader_options,
        ),
        validation=DataLoader(
            dataset=validation_subset,
            batch_size=config.evaluation_batch_size,
            shuffle=False,
            **loader_options,
        ),
        test=DataLoader(
            dataset=test_dataset,
            batch_size=config.evaluation_batch_size,
            shuffle=False,
            **loader_options,
        ),
        split=split,
    )


def train_classifier_epoch(
    model: MNISTClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> ClassificationMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = nn.functional.cross_entropy(output.logits, labels)
        loss.backward()
        optimizer.step()

        # some bookkeeping
        batch_size = labels.shape[0]
        total_examples += batch_size
        total_correct += int((output.logits.argmax(dim=1) == labels).sum())
        total_loss += float(loss.detach()) * batch_size

    return ClassificationMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


@torch.inference_mode()
def evaluate_classifier(
    model: MNISTClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    confusion = torch.zeros(MNIST_NUM_CLASSES, MNIST_NUM_CLASSES, dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = nn.functional.cross_entropy(output.logits, labels)
        predictions = output.logits.argmax(dim=1)

        batch_size = labels.shape[0]
        total_examples += batch_size
        total_correct += int((output.logits.argmax(dim=1) == labels).sum())
        total_loss += float(loss.detach()) * batch_size
        confusion += confusion_matrix_from_predictions(predictions, labels)

    return EvaluationResult(
        metrics=ClassificationMetrics(
            loss=total_loss / total_examples,
            accuracy=total_correct / total_examples,
        ),
        confusion_matrix=confusion,
    )


def train_mnist_classifier(
    config: MNISTClassifierTrainingConfig, *, device: torch.device | str = "cpu"
) -> MNISTClassifierTrainingResult:

    train_device = torch.device(device)
    seed_everything(config.training_seed)
    loaders = build_mnist_classifier_loaders(config, device=train_device)

    model = MNISTClassifier(
        feature_dim=config.feature_dim,
        dropout=config.dropout,
    ).to(train_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: list[EpochRecord] = []
    best_epoch = 0
    best_validation_accuracy = -1.0
    best_state_dict: dict[str, Tensor] | None = None

    for epoch in range(1, config.num_epochs + 1):
        training_metrics = train_classifier_epoch(
            model,
            loaders.training,
            optimizer,
            device=train_device,
        )
        validation_result = evaluate_classifier(
            model,
            loaders.validation,
            device=train_device,
        )
        validation_metrics = validation_result.metrics
        history.append(
            EpochRecord(
                epoch=epoch,
                training=training_metrics,
                validation=validation_metrics,
            )
        )

        # print metrics to stdout
        print(
            f"epoch={epoch:02d} "
            f"train_loss={training_metrics.loss:.4f} "
            f"train_accuracy={training_metrics.accuracy:.4f} "
            f"validation_loss={validation_metrics.loss:.4f} "
            f"validation_accuracy={validation_metrics.accuracy:.4f}"
        )

        # update best epoch numbers
        if validation_metrics.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_metrics.accuracy
            best_epoch = epoch
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    assert best_state_dict is not None
    model.load_state_dict(best_state_dict)
    test_result = evaluate_classifier(
        model,
        loaders.test,
        device=train_device,
    )

    return MNISTClassifierTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_accuracy=best_validation_accuracy,
        test=test_result,
        split=loaders.split,
    )
