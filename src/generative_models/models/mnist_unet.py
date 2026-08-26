from __future__ import annotations
from torch import nn, Tensor
import torch

from generative_models.models.continuous_time import ContinuousTimeEmbedding


class TimeConditionedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        conditioning_dim: int,
        num_groups: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_layers = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conditioning_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(conditioning_dim, out_channels)
        )
        self.output_layers = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, features: Tensor, conditioning: Tensor) -> Tensor:
        hidden = self.input_layers(features)
        condition_bias = self.conditioning_projection(conditioning)
        hidden = hidden + condition_bias[:, :, None, None]
        hidden = self.output_layers(hidden)
        return self.shortcut(features) + hidden


class MNISTUNet(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 32,
        time_embedding_dim: int = 64,
        num_classes: int = 10,
        num_groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels_28 = base_channels
        channels_14 = base_channels * 2
        channels_7 = base_channels * 4
        conditioning_dim = base_channels * 4
        self.conditioning_dim = conditioning_dim
        self.num_classes = num_classes

        # time embedding
        self.time_embedding = nn.Sequential(
            ContinuousTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )

        # class embedding
        self.class_embedding = nn.Embedding(
            num_classes,
            conditioning_dim
        )

        self.input_projection = nn.Conv2d(
            1,
            channels_28,
            kernel_size=3,
            padding=1,
        )
        self.encoder_28 = TimeConditionedResidualBlock(
            channels_28,
            channels_28,
            conditioning_dim=conditioning_dim,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.downsample_14 = nn.Conv2d(
            channels_28,
            channels_14,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.encoder_14 = TimeConditionedResidualBlock(
            channels_14,
            channels_14,
            conditioning_dim=conditioning_dim,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.downsample_7 = nn.Conv2d(
            channels_14,
            channels_7,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bottleneck_1 = TimeConditionedResidualBlock(
            channels_7,
            channels_7,
            conditioning_dim=conditioning_dim,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.bottleneck_2 = TimeConditionedResidualBlock(
            channels_7,
            channels_7,
            conditioning_dim=conditioning_dim,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.upsample_14 = nn.ConvTranspose2d(
            channels_7, channels_14, kernel_size=4, stride=2, padding=1
        )
        self.decoder_14 = TimeConditionedResidualBlock(
            channels_14 * 2,
            channels_14,
            conditioning_dim=conditioning_dim,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.upsample_28 = nn.ConvTranspose2d(
            channels_14,
            channels_28,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.decoder_28 = TimeConditionedResidualBlock(
            channels_28 * 2,
            channels_28,
            conditioning_dim=conditioning_dim,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(num_groups, channels_28),
            nn.SiLU(),
            nn.Conv2d(channels_28, 1, kernel_size=3, padding=1),
        )

    def forward(self, images: Tensor, times: Tensor, labels: Tensor) -> Tensor:
        # both labels and times should just have shape [B]
        if labels.shape != times.shape:
            raise ValueError(f"got labels={tuple(labels.shape)} and times={tuple(times.shape)}")
        # labels are indices and not continuous measurements
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")

        times = times.to(device=images.device, dtype=images.dtype)
        labels = labels.to(device=images.device)

        # combine conditioning -> time + class conditioning
        time_conditioning = self.time_embedding(times)
        class_conditioning = self.class_embedding(labels)
        # c = c_t + c_y
        conditioning = time_conditioning + class_conditioning

        features_28 = self.input_projection(images)
        skip_28 = self.encoder_28(features_28, conditioning)

        features_14 = self.downsample_14(skip_28)
        skip_14 = self.encoder_14(features_14, conditioning)

        features_7 = self.downsample_7(skip_14)
        features_7 = self.bottleneck_1(features_7, conditioning)
        features_7 = self.bottleneck_2(features_7, conditioning)

        decoded_14 = self.upsample_14(features_7)
        decoded_14 = torch.cat([decoded_14, skip_14], dim=1)
        decoded_14 = self.decoder_14(decoded_14, conditioning)

        decoded_28 = self.upsample_28(decoded_14)
        decoded_28 = torch.cat([decoded_28, skip_28], dim=1)
        decoded_28 = self.decoder_28(decoded_28, conditioning)

        return self.output_projection(decoded_28)
