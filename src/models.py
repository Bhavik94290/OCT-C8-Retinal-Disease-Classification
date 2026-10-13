import torch
import torch.nn as nn
from torchvision import models


NUM_CLASSES = 8


class ResNet50(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class VGG19(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.model = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT
        )

        in_features = self.model.classifier[0].in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class DenseNet121(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        in_features = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))

        attention = self.sigmoid(avg + max_)

        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=7,
            padding=3,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat([avg, max_], dim=1)

        attention = self.sigmoid(self.conv(combined))

        return x * attention


class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, in_channels=1024, out_channels=256):
        super().__init__()

        c = out_channels // 3

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, c, 5, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(c * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)

        x = torch.cat([b1, b3, b5], dim=1)

        x = self.fusion(x)

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class ProposedOCTNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        backbone = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        self.features = backbone.features

        self.multi_scale = MultiScaleAttentionBlock(
            1024,
            256
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)

        x = self.multi_scale(x)

        x = self.pool(x)

        return self.classifier(x)


def get_model(name):
    name = name.lower()

    if name == "resnet50":
        return ResNet50()

    if name == "vgg19":
        return VGG19()

    if name == "densenet121":
        return DenseNet121()

    if name == "proposed":
        return ProposedOCTNet()

    raise ValueError(
        f"Unknown model: {name}"
    )
