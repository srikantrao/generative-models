import matplotlib.pyplot as plt
import torch
from torch._inductor.ir import NoneLayout

from generative_models.data.toy import sample_labeled_gaussian_mixture
from generative_models.diffusion.forward import q_sample
from generative_models.diffusion.schedules import (
    build_diffusion_schedule,
    linear_beta_schedule,
)
from generative_models.viz.points import (
    plot_labeled_points,
    save_figure,
    use_clean_style,
)


def main() -> None:
    generator = torch.Generator().manual_seed(42)
    batch = sample_labeled_gaussian_mixture(1000, generator=generator)
    schedule = build_diffusion_schedule(linear_beta_schedule(1_000))
    noise = torch.randn(
        batch.x.shape,
        generator=generator,
        device=batch.x.device,
        dtype=batch.x.dtype,
    )

    timestep_indices = [0, 49, 199, 499, 999]
    use_clean_style()
    fig, axes = plt.subplots(
        1,
        len(timestep_indices),
        figsize=(18, 4),
        sharex=True,
        sharey=True,
    )

    for ax, timestep in zip(axes, timestep_indices, strict=True):
        timesteps = torch.full(
            (batch.x.shape[0],),
            timestep,
            dtype=torch.long,
        )
        sample = q_sample(schedule, batch.x, timesteps, noise=noise)
        _ = plot_labeled_points(
            ax,
            sample.xt,
            labels=batch.y,
            title=f"t = {timestep}",
            class_names=["mode 0", "mode 1", "mode 2"],
            point_size=10.0,
            alpha=0.55,
        )
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    path = save_figure(fig, "figures/forward_process.png")
    plt.close(fig)

    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
