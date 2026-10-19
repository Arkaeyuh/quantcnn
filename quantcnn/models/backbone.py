"""CNN feature extractor with global pooling and bottleneck projection."""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Lightweight convolutional backbone for MNIST."""

    def __init__(self, bottleneck_dim: int, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = self.gap(h).flatten(1)
        return self.proj(h)


class CifarCNN(nn.Module):
    """Small CNN backbone for RGB 32×32 inputs (e.g., CIFAR-10)."""

    def __init__(self, bottleneck_dim: int, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = self.gap(h).flatten(1)
        return self.proj(h)


class PretrainedResNet18(nn.Module):
    """ImageNet-pretrained ResNet-18 fine-tuned as the backbone for CIFAR-10.

    Inputs arrive CIFAR-normalized at 32x32; they are mapped back to pixel space,
    re-normalized with ImageNet statistics and upsampled to ``resolution`` before the
    network. The classifier is replaced by a linear projection to the bottleneck.
    """

    CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, bottleneck_dim: int, resolution: int = 128) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Linear(512, bottleneck_dim)
        self.resolution = resolution
        shape = (1, 3, 1, 1)
        self.register_buffer("c_mean", torch.tensor(self.CIFAR_MEAN).view(shape), persistent=False)
        self.register_buffer("c_std", torch.tensor(self.CIFAR_STD).view(shape), persistent=False)
        self.register_buffer("i_mean", torch.tensor(self.IMAGENET_MEAN).view(shape), persistent=False)
        self.register_buffer("i_std", torch.tensor(self.IMAGENET_STD).view(shape), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.c_std + self.c_mean - self.i_mean) / self.i_std
        x = nn.functional.interpolate(x, size=self.resolution, mode="bilinear", align_corners=False)
        return self.net(x)
