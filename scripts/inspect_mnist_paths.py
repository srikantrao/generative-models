from __future__ import annotations

import argparse
import json
import torch
import torchvision
import yaml
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from pathlib import Path

from torch import Tensor


from generative_models.data.mnist import (
    MNIST_NORMALIZATION,
    denormalize_mnist_image,
    index_set_sha256,
    load_mnist_split,
    make_mnist_evaluator_split,
    mnist_raw_file_sha256,
)
from generative_models.diffusion.forward import q_sample
from generative_models.diffusion.schedules import (
    build_diffusion_schedule,
    linear_beta_schedule,
)
from generative_models.flow_matching.paths import sample_linear_flow_path
from generative_models.viz.points import use_clean_style, save_figure


@dataclass(frozen=True)
class MNISTPathArtifactConfig:
    data_root: str
    download: bool
    image_index: int
    seed: int
    split_seed: int
    validation_size: int
    num_timesteps: int
    beta_start: float
    beta_end: float
    ddpm_timesteps: tuple[int, ...]
    flow_times: tuple[float, ...]

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
    ) -> MNISTPathArtifactConfig:
        required_keys = frozenset(
            {
                "data_root",
                "download",
                "image_index",
                "seed",
                "split_seed",
                "validation_size",
                "num_timesteps",
                "beta_start",
                "beta_end",
                "ddpm_timesteps",
                "flow_times",
            }
        )
        missing_keys = required_keys.difference(values)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"config is missing required keys: {missing}")

        data_root = values["data_root"]
        download = values["download"]
        raw_ddpm_timesteps = values["ddpm_timesteps"]
        raw_flow_times = values["flow_times"]

        if not isinstance(data_root, str):
            raise ValueError("data_root must be a string")
        if not isinstance(download, bool):
            raise ValueError("download must be a boolean")
        if not isinstance(raw_ddpm_timesteps, list):
            raise ValueError("ddpm_timesteps must be a list")
        if not isinstance(raw_flow_times, list):
            raise ValueError("flow_times must be a list")

        config = cls(
            data_root=data_root,
            download=download,
            image_index=int(values["image_index"]),
            seed=int(values["seed"]),
            split_seed=int(values["split_seed"]),
            validation_size=int(values["validation_size"]),
            num_timesteps=int(values["num_timesteps"]),
            beta_start=float(values["beta_start"]),
            beta_end=float(values["beta_end"]),
            ddpm_timesteps=tuple(int(timestep) for timestep in raw_ddpm_timesteps),
            flow_times=tuple(float(time) for time in raw_flow_times),
        )
        return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mnist_data.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/mnist_paths"),
    )
    return parser.parse_args()


def plot_image(
    ax: Axes,
    image: Tensor,
    *,
    title: str,
) -> None:
    display = denormalize_mnist_image(image).detach().cpu()
    ax.imshow(
        display.squeeze(0).numpy(),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def load_config(path: Path) -> MNISTPathArtifactConfig:
    raw_config = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw_config, Mapping):
        raise ValueError("config must contain a mapping")
    return MNISTPathArtifactConfig.from_mapping(raw_config)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_dataset = load_mnist_split(
        config.data_root, train=True, download=config.download
    )

    test_dataset = load_mnist_split(
        config.data_root, train=False, download=config.download
    )

    # choose one image based on index from config
    clean_image, label = train_dataset[config.image_index]
    data: Tensor = clean_image.unsqueeze(0)  # add the batch dimension
    generator = torch.Generator().manual_seed(config.seed)
    # generate noise that will be used for both DDPM and Flow-matching.
    source_noise = torch.randn(
        data.shape,
        generator=generator,
        dtype=data.dtype,
        device=data.device,
    )

    schedule = build_diffusion_schedule(
        linear_beta_schedule(
            config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            dtype=data.dtype,
        )
    )

    ## DDPM
    ddpm_images: list[Tensor] = []
    for timestep in config.ddpm_timesteps:
        timesteps = torch.tensor([timestep], dtype=torch.long)
        sample = q_sample(schedule, data, timesteps, noise=source_noise)
        ddpm_images.append(sample.xt[0])

    ## Flow matching
    flow_images: list[Tensor] = []
    for time in config.flow_times:
        times = torch.tensor([time], dtype=data.dtype)
        sample = sample_linear_flow_path(
            data,
            times=times,
            source_noise=source_noise,
        )
        flow_images.append(sample.xt[0])
    # at time t=0, this should equal source noise
    if not torch.allclose(flow_images[0], source_noise[0]):
        raise RuntimeError("flow time-zero endpoint does not equal source noise")
    # at time t=1, this should equal the data
    if not torch.allclose(flow_images[-1], clean_image):
        raise RuntimeError("flow time-one endpoint does not equal clean data")

    # Plotting
    use_clean_style()
    num_columns = len(config.ddpm_timesteps)
    fig, axes = plt.subplots(
        2,
        num_columns,
        figsize=(3.0 * num_columns, 6.0),
        squeeze=False,
    )

    for ax, image, timestep in zip(
        axes[0], ddpm_images, config.ddpm_timesteps, strict=True
    ):
        plot_image(ax, image, title=f"t = {timestep}")

    for ax, image, time in zip(axes[1], flow_images, config.flow_times, strict=True):
        plot_image(ax, image, title=f"t = {time:2f}")

    axes[0, 0].set_ylabel("DDPM\ndata -> noise")
    axes[1, 0].set_ylabel("Flow\nnoise -> data")
    fig.suptitle(
        f"MNIST probability paths | index={config.image_index} "
        f"label={label} source_seed={config.seed}"
    )
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.94))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = save_figure(fig, args.output_dir / "mnist_probability_paths.png")
    plt.close()

    evaluator_split = make_mnist_evaluator_split(
        num_examples=len(train_dataset),
        validation_size=config.validation_size,
        seed=config.split_seed,
    )

    summary = {
        "config": asdict(config),
        "dataset": {
            "class": "torchvision.datasets.MNIST",
            "root": config.data_root,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "raw_image_shape": list(train_dataset.data.shape[1:]),
            "raw_dtype": str(train_dataset.data.dtype),
            "raw_pixel_range": [
                int(train_dataset.data.min()),
                int(train_dataset.data.max()),
            ],
            "transformed_image_shape": list(clean_image.shape),
            "transformed_dtype": str(clean_image.dtype),
            "declared_normalization": MNIST_NORMALIZATION,
            "raw_file_sha256": mnist_raw_file_sha256(config.data_root),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        },
        "evaluator_split": {
            "seed": config.split_seed,
            "train_size": evaluator_split.train_indices.numel(),
            "validation_size": evaluator_split.validation_indices.numel(),
            "train_index_set_sha256": index_set_sha256(evaluator_split.train_indices),
            "validation_index_set_sha256": index_set_sha256(
                evaluator_split.validation_indices
            ),
        },
        "known_endpoints": {
            "training_image_index": config.image_index,
            "label": label,
            "source_distribution": "standard_normal",
            "source_seed": config.seed,
            "same_source_used_for_both_paths": True,
        },
        "ddpm_path": {
            "direction": "data_to_noise",
            "index_convention": "zero_based_0_to_K_minus_1",
            "formula": ("x_k=sqrt(alpha_bar_k)*x_0+sqrt(1-alpha_bar_k)*epsilon"),
            "timesteps": list(config.ddpm_timesteps),
            "selected_alpha_bars": [
                float(schedule.alpha_bars[timestep])
                for timestep in config.ddpm_timesteps
            ],
        },
        "flow_path": {
            "direction": "noise_to_data",
            "formula": "x_t=(1-t)*z+t*x_1",
            "target_velocity": "x_1-z",
            "times": list(config.flow_times),
            "time_zero_matches_source": True,
            "time_one_matches_data": True,
        },
        "figure_path": str(figure_path),
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {figure_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
