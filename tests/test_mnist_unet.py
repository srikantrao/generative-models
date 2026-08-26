import torch
import pytest
from generative_models.models.mnist_unet import MNISTUNet

def test_conditioned_mnist_unet_backpropagates_through_class_embedding() -> None:
    torch.manual_seed(42)
    model = MNISTUNet(
        base_channels=16,
        time_embedding_dim=32,
        dropout=0.0,
    )
    image = torch.randn(1, 1, 28, 28)
    # copy the same image over as the second one in the batch
    images = image.repeat(2, 1, 1, 1)
    times = torch.full((2,), 0.5) # [0.5, 0.5]
    labels = torch.tensor([2, 7], dtype=torch.long)

    predictions = model(images, times, labels)
    loss = predictions.square().mean()
    loss.backward()

    assert predictions.shape == images.shape
    assert predictions.dtype == images.dtype
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(loss)
    assert not torch.allclose(predictions[0], predictions[1])

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    class_gradient = model.class_embedding.weight.grad
    assert class_gradient is not None
    assert torch.count_nonzero(class_gradient[labels]).item() > 0


def test_conditioned_mnist_unet_rejects_broadcastable_label_shape() -> None:
    model = MNISTUNet(
        base_channels=16,
        time_embedding_dim=32,
        dropout=0.0,
    )
    images = torch.randn(2, 1, 28, 28)
    times = torch.full((2,), 0.5)
    labels = torch.tensor([[2], [7]], dtype=torch.long)

    with pytest.raises(ValueError, match="labels must have the same"):
        model(images, times, labels)
