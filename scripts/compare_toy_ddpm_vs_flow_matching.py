from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from generative_models.data.toy import sample_labeled_gaussian_mixture
from generative_models.diffusion.samplers import sample_ddpm
from generative_models.diffusion.schedules import (
    build_diffusion_schedule,
    linear_beta_schedule,
)
from generative_models.experiments.toy_checkpoints import (
    LoadedToyDDPMCheckpoint,
    LoadedToyFlowCheckpoint,
    load_toy_ddpm_checkpoint,
    load_toy_flow_checkpoint,
)
from generative_models.flow_matching.ode_samplers import (
    ODESolver,
    sample_flow_ode,
)
from generative_models.training.toy_ddpm import ToyDDPMTrainingConfig
from generative_models.training.toy_flow_matching import (
    ToyFlowMatchingTrainingConfig,
)
from generative_models.viz.points import (
    plot_labeled_points,
    save_figure,
    use_clean_style,
)

MATCHED_TRAINING_FIELDS = (
    "num_steps",
    "batch_size",
    "learning_rate",
    "hidden_dim",
    "time_embedding_dim",
    "num_hidden_layers",
    "seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddpm-checkpoint",
        type=Path,
        default=Path("runs/toy_ddpm/checkpoint.pt"),
    )
    parser.add_argument(
        "--flow-checkpoint",
        type=Path,
        default=Path("runs/toy_flow_matching/checkpoint.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/toy_ddpm_vs_flow"),
    )
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--flow-num-steps", type=int, default=100)
    parser.add_argument(
        "--flow-solver",
        choices=("euler", "heun"),
        default="heun",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def matched_training_controls(
    ddpm_config: ToyDDPMTrainingConfig,
    flow_config: ToyFlowMatchingTrainingConfig,
) -> dict[str, object]:
    matched: dict[str, object] = {}
    mismatches: list[str] = []

    for field_name in MATCHED_TRAINING_FIELDS:
        ddpm_value = getattr(ddpm_config, field_name)
        flow_value = getattr(flow_config, field_name)
        if ddpm_value != flow_value:
            mismatches.append(f"{field_name}: ddpm={ddpm_value!r}, flow={flow_value!r}")
        else:
            matched[field_name] = ddpm_value

    if ddpm_config.data_spec != flow_config.data_spec:
        mismatches.append("data_spec differs between DDPM and flow checkpoints")
    else:
        matched["data_spec"] = asdict(ddpm_config.data_spec)

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"training controls are not matched: {details}")
    return matched


def validate_matched_parameter_count(
    ddpm: LoadedToyDDPMCheckpoint,
    flow: LoadedToyFlowCheckpoint,
) -> int:
    if ddpm.num_parameters != flow.num_parameters:
        raise ValueError(
            "model parameter counts are not matched: "
            f"ddpm={ddpm.num_parameters}, flow={flow.num_parameters}"
        )
    return ddpm.num_parameters


def flow_nfe(solver: ODESolver, num_steps: int) -> int:
    evaluations_per_step = 1 if solver == "euler" else 2
    return evaluations_per_step * num_steps


def run_comparison_artifact(
    *,
    ddpm_checkpoint_path: Path,
    flow_checkpoint_path: Path,
    output_dir: Path,
    num_samples: int,
    flow_num_steps: int,
    flow_solver: ODESolver,
    seed: int,
    device: torch.device | str,
) -> dict[str, object]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if flow_num_steps <= 0:
        raise ValueError("flow_num_steps must be positive")
    if flow_solver not in ("euler", "heun"):
        raise ValueError("flow_solver must be 'euler' or 'heun'")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    sample_device = torch.device(device)
    ddpm = load_toy_ddpm_checkpoint(
        ddpm_checkpoint_path,
        device=sample_device,
    )
    flow = load_toy_flow_checkpoint(
        flow_checkpoint_path,
        device=sample_device,
    )
    controls = matched_training_controls(ddpm.config, flow.config)
    num_parameters = validate_matched_parameter_count(ddpm, flow)

    ddpm_dtype = next(ddpm.model.parameters()).dtype
    flow_dtype = next(flow.model.parameters()).dtype
    if ddpm_dtype != flow_dtype:
        raise ValueError(
            f"model dtypes are not matched: ddpm={ddpm_dtype}, flow={flow_dtype}"
        )

    schedule = build_diffusion_schedule(
        linear_beta_schedule(
            ddpm.config.num_timesteps,
            beta_start=ddpm.config.beta_start,
            beta_end=ddpm.config.beta_end,
            device=sample_device,
        )
    )

    ddpm_sampler_seed = seed
    flow_source_seed = seed + 1
    reference_seed = seed + 2

    ddpm_generator = torch.Generator(device=sample_device).manual_seed(
        ddpm_sampler_seed
    )
    flow_generator = torch.Generator(device=sample_device).manual_seed(flow_source_seed)
    reference_generator = torch.Generator(device=sample_device).manual_seed(
        reference_seed
    )

    ddpm_output = sample_ddpm(
        ddpm.model,
        schedule,
        num_samples=num_samples,
        sample_shape=(2,),
        device=sample_device,
        generator=ddpm_generator,
    )
    flow_output = sample_flow_ode(
        flow.model,
        num_samples=num_samples,
        sample_shape=(2,),
        num_steps=flow_num_steps,
        solver=flow_solver,
        device=sample_device,
        dtype=flow_dtype,
        generator=flow_generator,
    )
    centers, class_probs = ddpm.config.data_spec.to_tensors(device=sample_device)
    reference = sample_labeled_gaussian_mixture(
        num_samples,
        centers=centers,
        std=ddpm.config.data_spec.std,
        class_probs=class_probs,
        generator=reference_generator,
        device=sample_device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "comparison.png"
    summary_path = output_dir / "summary.json"

    use_clean_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        sharex=True,
        sharey=True,
    )
    plot_labeled_points(
        axes[0],
        reference.x,
        labels=reference.y,
        title="Real Toy Data",
        class_names=["mode 0", "mode 1", "mode 2"],
        point_size=10.0,
        alpha=0.6,
    )
    plot_labeled_points(
        axes[1],
        ddpm_output.samples,
        title="DDPM Samples",
        point_size=10.0,
        alpha=0.6,
    )
    plot_labeled_points(
        axes[2],
        flow_output.samples,
        title=f"Flow Samples ({flow_solver})",
        point_size=10.0,
        alpha=0.6,
    )
    for axis in axes:
        axis.set_xlim(-4, 4)
        axis.set_ylim(-4, 4)

    save_figure(figure, figure_path)
    plt.close(figure)

    summary: dict[str, object] = {
        "artifact_type": "toy_ddpm_vs_flow_qualitative_comparison",
        "comparison_scope": "qualitative_implementation_check",
        "num_samples": num_samples,
        "device": str(sample_device),
        "dtype": str(ddpm_dtype),
        "paired_initial_sources": False,
        "matched_training_controls": controls,
        "num_model_parameters_each": num_parameters,
        "checkpoints": {
            "ddpm": {
                "path": str(ddpm_checkpoint_path),
                "sha256": ddpm.sha256,
                "config": asdict(ddpm.config),
            },
            "flow": {
                "path": str(flow_checkpoint_path),
                "sha256": flow.sha256,
                "config": asdict(flow.config),
            },
        },
        "sampling": {
            "ddpm": {
                "sampler": "ancestral",
                "num_steps": schedule.num_timesteps,
                "nfe": schedule.num_timesteps,
                "sampler_seed": ddpm_sampler_seed,
                "stochastic_after_initialization": True,
                "time_direction": "num_timesteps-1_to_0",
            },
            "flow": {
                "sampler": flow_solver,
                "num_steps": flow_num_steps,
                "nfe": flow_nfe(flow_solver, flow_num_steps),
                "source_seed": flow_source_seed,
                "stochastic_after_initialization": False,
                "time_direction": "0_to_1",
            },
            "reference_seed": reference_seed,
        },
        "artifacts": {
            "figure": str(figure_path),
            "summary": str(summary_path),
        },
        "claim_boundary": [
            "The figure verifies that both sampling paths execute end to end.",
            "The figure is not a quantitative distribution comparison.",
            "No conclusion about model-family superiority is supported.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_comparison_artifact(
        ddpm_checkpoint_path=args.ddpm_checkpoint,
        flow_checkpoint_path=args.flow_checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        flow_num_steps=args.flow_num_steps,
        flow_solver=args.flow_solver,
        seed=args.seed,
        device=args.device,
    )
    print(f"Wrote {summary['artifacts']['figure']}")
    print(f"Wrote {summary['artifacts']['summary']}")


if __name__ == "__main__":
    main()


"""
uv run --group ml python scripts/compare_toy_ddpm_vs_flow_matching.py \
  --ddpm-checkpoint runs/toy_ddpm/checkpoint.pt \
  --flow-checkpoint runs/toy_flow_matching/checkpoint.pt \
  --output-dir runs/toy_ddpm_vs_flow \
  --num-samples 1000 \
  --flow-num-steps 100 \
  --flow-solver heun \
  --seed 123 \
  --device cpu
"""
