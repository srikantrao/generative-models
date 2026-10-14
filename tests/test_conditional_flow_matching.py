import pytest
import torch

from generative_models.flow_matching.conditional import (
    sample_conditional_flow_ode
)

class LabelVelocity:
    def __call__(
        self,
        state: torch.Tensor,
        times: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        del times
        values = labels.to(dtype=state.dtype)
        return values[:, None, None, None].expand_as(state)

@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_conditional_flow_solver_holds_labels_fixed(solver: str) -> None:
    source = torch.zeros(2, 1, 2, 2)
    labels = torch.tensor([1, 2])

    output = sample_conditional_flow_ode(
        LabelVelocity(),
        labels,
        source,
        num_steps=4,
        solver=solver,
    )

    expected = labels.to(torch.float32)[:, None, None, None].expand_as(source)
    torch.testing.assert_close(output.samples, expected)
