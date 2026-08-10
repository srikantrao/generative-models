import torch

from generative_models.models.mnist_unet import MNISTUNet

def test_mnist_unet_preserves_image_contract_and_backpropagates() -> None:
    torch.manual_seed(42)
    model = MNISTUNet(
        base_channels=16,
        time_embedding_dim=32,
        dropout=0.0
    )

    images = torch.randn(2, 1, 28, 28)
    times = torch.tensor([0.0, 0.75], dtype=torch.float32)

    predictions = model(images, times)
    loss = predictions.square().mean()
    loss.backward()

    assert predictions.shape == images.shape
    assert predictions.dtype == images.dtype
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(loss)

    for parameter in model.parameters():
        assert parameter.grad is not None;
        assert torch.isfinite(parameter.grad).all()
