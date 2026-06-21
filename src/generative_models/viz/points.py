from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

ArrayLike = Tensor | np.ndarray


def use_clean_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _to_numpy(array: ArrayLike) -> np.ndarray:
    if isinstance(array, Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _validate_points(points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape [num_points, 2], got {points.shape}")


def plot_labeled_points(
    ax: Axes,
    points: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    title: str,
    class_names: list[str] | None = None,
    point_size: float = 16.0,
    alpha: float = 0.75,
) -> Axes:
    points_np = _to_numpy(points)
    _validate_points(points_np)

    if labels is None:
        ax.scatter(
            points_np[:, 0],
            points_np[:, 1],
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
    else:
        labels_np = _to_numpy(labels).astype(np.int64)
        if labels_np.shape != (points_np.shape[0],):
            raise ValueError(
                f"labels must have shape [{points_np.shape[0]}], got {labels_np.shape}"
            )

        for label in np.unique(labels_np):
            mask = labels_np == label
            name = str(label)
            if class_names is not None and 0 <= label < len(class_names):
                name = class_names[label]

            ax.scatter(
                points_np[mask, 0],
                points_np[mask, 1],
                s=point_size,
                alpha=alpha,
                linewidths=0,
                label=name,
            )
        ax.legend(frameon=False)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    return ax


def save_figure(fig: Figure, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return output_path
