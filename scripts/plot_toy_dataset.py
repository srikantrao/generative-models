import matplotlib.pyplot as plt
import torch

from generative_models.data.toy import sample_labeled_gaussian_mixture
from generative_models.viz.points import (
    plot_labeled_points,
    save_figure,
    use_clean_style,
)


def main() -> None:
    generator = torch.Generator().manual_seed(42)
    batch = sample_labeled_gaussian_mixture(
        1_000,
        generator=generator,
    )

    use_clean_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_labeled_points(
        ax,
        batch.x,
        labels=batch.y,
        title="Labeled 2D Gaussian Mixture",
        class_names=["mode 0", "mode 1", "mode 2"],
    )
    path = save_figure(fig, "figures/toy_dataset.png")
    plt.close(fig)

    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
