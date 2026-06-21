import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

from generative_models.viz.points import plot_labeled_points, save_figure


def test_plot_labeled_points_accepts_tensor_inputs() -> None:
    points = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0], dtype=torch.int64)

    fig, ax = plt.subplots()
    returned_ax = plot_labeled_points(
        ax,
        points,
        labels=labels,
        title="test",
        class_names=["left", "right"],
    )

    assert returned_ax is ax
    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_labeled_points_accepts_unlabeled_points() -> None:
    points = torch.zeros(4, 2)

    fig, ax = plt.subplots()
    plot_labeled_points(ax, points, title="unlabeled")

    assert len(ax.collections) == 1
    plt.close(fig)


def test_plot_labeled_points_rejects_invalid_point_shape() -> None:
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="points must have shape"):
        plot_labeled_points(ax, torch.zeros(4, 3), title="bad")

    plt.close(fig)


def test_plot_labeled_points_rejects_label_length_mismatch() -> None:
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="labels must have shape"):
        plot_labeled_points(
            ax,
            torch.zeros(4, 2),
            labels=torch.zeros(3, dtype=torch.int64),
            title="bad labels",
        )

    plt.close(fig)


def test_save_figure_creates_parent_directory(tmp_path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    output_path = save_figure(fig, tmp_path / "nested" / "figure.png")

    assert output_path.exists()
    assert output_path.name == "figure.png"
    plt.close(fig)
