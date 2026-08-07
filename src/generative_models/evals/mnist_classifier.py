from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

@dataclass(frozen=True)
class MNISTClassifierOutput:
    logits: Tensor
    features: Tensor

class MNISTClassifier(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.feature_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, images: Tensor) -> MNISTClassifierOutput:
        encoded = self.encoder(images)
        features: Tensor = self.feature_projection(encoded)
        logits: Tensor = self.classifier(self.dropout(features))
        return MNISTClassifierOutput(logits=logits, features=features)

def freeze_mnist_classifier(model: MNISTClassifier) -> MNISTClassifier:
    model.eval()
    model.requires_grad_(False)
    return model
