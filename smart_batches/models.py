"""Models. SmallCNN is deliberately BatchNorm-free so the exact per-sample
gradient scorer (torch.func.vmap) is well-defined on it."""

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, img_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        spatial = img_size // 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * spatial * spatial, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str, in_channels: int, num_classes: int, img_size: int = 28) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(in_channels, num_classes, img_size)
    if name == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(weights=None, num_classes=num_classes)
        # CIFAR stem: 3x3 stride-1 conv, no max-pool (32x32 inputs)
        model.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown model {name!r}")
