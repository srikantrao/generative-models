import torch

from generative_models.diffusion.conditional import (
    conditional_epsilon_loss,
    sample_conditional_ddpm,
)
from generative_models.diffusion.schedules import (
    build_diffusion_schedule,
    linear_beta_schedule,
)


class RecordingPredictor:
    def __init__(self) -> None:
        self.times: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None

    def __call__(
        self,
        images: torch.Tensor,
        times: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        self.times = times
        self.labels = labels
        return torch.zeros_like(images)


def test_conditional_ddpm_maps_schedule_endpoints_to_unit_time() -> None:
    model = RecordingPredictor()
    schedule = build_diffusion_schedule(linear_beta_schedule(4))
    images = torch.zeros(2, 1, 4, 4)
    labels = torch.tensor([2, 7])
    timesteps = torch.tensor([0, 3])
    noise = torch.ones_like(images)

    output = conditional_epsilon_loss(
        model,
        schedule,
        images,
        labels,
        timesteps=timesteps,
        noise=noise,
    )

    torch.testing.assert_close(model.times, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(model.labels, labels)
    torch.testing.assert_close(output.loss, torch.tensor(1.0))


def test_conditional_sampler_replays_noise_and_preserves_labels() -> None:
    schedule = build_diffusion_schedule(linear_beta_schedule(4))
    labels = torch.tensor([2, 7])
    initial_noise = torch.ones(2, 1, 4, 4)
    seen_times = []

    def predictor(images, times, requested_labels):
        torch.testing.assert_close(requested_labels, labels)
        seen_times.append(times.clone())
        return torch.zeros_like(images)

    def sample(seed):
        return sample_conditional_ddpm(
            predictor,
            schedule,
            labels,
            initial_noise,
            ancestral_generator=torch.Generator().manual_seed(seed),
            return_trajectory=True,
        )

    first = sample(10)
    torch.testing.assert_close(
        torch.stack(seen_times),
        torch.tensor([1.0, 2 / 3, 1 / 3, 0.0])[:, None].expand(4, 2),
    )
    replay = sample(10)
    changed_seed = sample(11)
    torch.testing.assert_close(first.trajectory, replay.trajectory, rtol=0, atol=0)
    assert not torch.equal(first.samples, changed_seed.samples)
    assert first.trajectory.shape == (5, 2, 1, 4, 4)
    torch.testing.assert_close(first.trajectory[0], initial_noise)
    # With zero predicted noise, the final transition is only the mean rescaling.
    torch.testing.assert_close(
        first.samples, first.trajectory[-2] / schedule.alphas[0].sqrt()
    )
