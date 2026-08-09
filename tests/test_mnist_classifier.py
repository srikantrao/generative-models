import torch

from generative_models.training.mnist_classifier import (
    confusion_matrix_from_predictions,
)


def test_confusion_matrix_uses_true_rows_and_predicted_columns() -> None:
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 1, 0, 2])

    confusion = confusion_matrix_from_predictions(predictions, targets, num_classes=3)

    expected = torch.tensor(
        [
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1],
        ]
    )
    torch.testing.assert_close(confusion, expected)
