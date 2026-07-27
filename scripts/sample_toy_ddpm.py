from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from generative_models.data.toy import ToyBatch, sample_labeled_gaussian_mixture
from generative_models.diffusion.samplers import sample_ddpm
from generative_models.diffusion.schedules import (
    build_diffusion_schedule,
    linear_beta_schedule,
)
from generative_models.experiments.toy_checkpoints import (
    load_toy_ddpm_checkpoint,
)
from generative_models.training.toy_ddpm import ToyDDPMTrainingConfig
from generative_models.viz.points import (
    plot_labeled_points,
    save_figure,
    use_clean_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/toy_ddpm/checkpoint.pt"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("figures/toy_ddpm_samples.png"),
    )
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def sample_reference_data(
    config: ToyDDPMTrainingConfig,
    *,
    num_samples: int,
    device: torch.device,
    generator: torch.Generator,
) -> ToyBatch:
    centers, class_probs = config.data_spec.to_tensors(device=device)
    return sample_labeled_gaussian_mixture(
        num_samples,
        centers=centers,
        class_probs=class_probs,
        std=config.data_spec.std,
        device=device,
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loaded = load_toy_ddpm_checkpoint(args.checkpoint, device=device)
    model, config = loaded.model, loaded.config
    schedule = build_diffusion_schedule(
        linear_beta_schedule(
            config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=device,
        )
    )

    sample_generator = torch.Generator(device=device).manual_seed(args.seed)
    generated = sample_ddpm(
        model,
        schedule,
        num_samples=args.num_samples,
        sample_shape=(2,),
        device=device,
        generator=sample_generator,
    )

    data_generator = torch.Generator(device=device).manual_seed(args.seed)
    real_batch = sample_reference_data(
        config,
        num_samples=args.num_samples,
        device=device,
        generator=data_generator,
    )

    use_clean_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    plot_labeled_points(
        axes[0],
        real_batch.x,
        labels=real_batch.y,
        title="Real Toy Data",
        class_names=[f"mode {index}" for index in range(len(config.data_spec.centers))],
        point_size=10.0,
        alpha=0.6,
    )
    plot_labeled_points(
        axes[1],
        generated.samples,
        title="DDPM Samples",
        point_size=10.0,
        alpha=0.6,
    )
    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    output_path = save_figure(fig, args.output_path)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
