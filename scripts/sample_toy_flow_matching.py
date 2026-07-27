from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from generative_models.data.toy import sample_labeled_gaussian_mixture
from generative_models.flow_matching.ode_samplers import ODESolver, sample_flow_ode
from generative_models.experiments.toy_checkpoints import (
    load_toy_flow_checkpoint,
)
from generative_models.viz.points import (
    plot_labeled_points,
    save_figure,
    use_clean_style,
)

REQUIRED_CHECKPOINT_KEYS = frozenset({"config", "model_state_dict"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/toy_flow_matching/checkpoint.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/toy_flow_samples"),
    )
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument(
        "--solver",
        choices=("euler", "heun"),
        default="heun",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    with path.open("rb") as checkpoint_file:
        return hashlib.file_digest(checkpoint_file, "sha256").hexdigest()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def neural_function_evaluations(solver: ODESolver, num_steps: int) -> int:
    calls_per_step = 1 if solver == "euler" else 2
    return calls_per_step * num_steps


def run_sampling_artifact(
    checkpoint_path: Path,
    output_dir: Path,
    num_samples: int,
    num_steps: int,
    solver: ODESolver,
    seed: int,
    device: torch.device | str,
) -> dict[str, object]:

    sample_device = torch.device(device)
    loaded = load_toy_flow_checkpoint(checkpoint_path, device=sample_device)
    model = loaded.model
    training_config = loaded.config
    model_dtype = next(model.parameters()).dtype

    source_generator = torch.Generator(device=sample_device).manual_seed(seed)
    generated = sample_flow_ode(
        model,
        num_samples=num_samples,
        sample_shape=(2,),
        num_steps=num_steps,
        solver=solver,
        device=device,
        generator=source_generator,
    )

    reference_seed = seed + 1
    reference_generator = torch.Generator(device=sample_device).manual_seed(
        reference_seed
    )
    centers, class_probs = training_config.data_spec.to_tensors(device=sample_device)
    real_batch = sample_labeled_gaussian_mixture(
        num_samples,
        centers=centers,
        std=training_config.data_spec.std,
        class_probs=class_probs,
        generator=reference_generator,
        device=sample_device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "samples.png"
    summary_path = output_dir / "summary.json"

    use_clean_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    plot_labeled_points(
        axes[0],
        real_batch.x,
        labels=real_batch.y,
        title="Real Toy Data",
        class_names=[
            f"mode {index}" for index in range(len(training_config.data_spec.centers))
        ],
        point_size=10.0,
        alpha=0.6,
    )
    plot_labeled_points(
        axes[1],
        generated.samples,
        title=f"Flow Matching Samples ({solver.title()})",
        point_size=10.0,
        alpha=0.6,
    )
    for axis in axes:
        axis.set_xlim(-4, 4)
        axis.set_ylim(-4, 4)

    save_figure(fig, figure_path)
    plt.close(fig)

    summary: dict[str, object] = {
        "artifact_type": "toy_flow_sampling",
        "artifact_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "training_config": asdict(training_config),
        "model": {
            "class_name": type(model).__name__,
            "num_parameters": count_parameters(model),
        },
        "sampling": {
            "source_seed": seed,
            "reference_seed": reference_seed,
            "device": str(sample_device),
            "dtype": str(model_dtype).removeprefix("torch."),
            "num_samples": num_samples,
            "num_steps": num_steps,
            "solver": solver,
            "nfe": neural_function_evaluations(solver, num_steps),
        },
        "artifacts": {
            "figure_path": str(figure_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    run_sampling_artifact(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        solver=args.solver,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
