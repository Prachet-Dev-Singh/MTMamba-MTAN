"""
ResNet Backbone Wrapper for MTMamba.

This module provides a multi-scale ResNet backbone (ResNet-18, ResNet-34, ResNet-50)
that exposes the exact interface required by the MTMamba multi-task decoder:
  - backbone.num_features: list of channels at each of the 4 stages
  - backbone(x): list of 4 feature maps [f1, f2, f3, f4] in standard PyTorch NCHW format:
      Stage 1 (f1): (B, C1, H/4,  W/4)
      Stage 2 (f2): (B, C2, H/8,  W/8)
      Stage 3 (f3): (B, C3, H/16, W/16)
      Stage 4 (f4): (B, C4, H/32, W/32)

Pretrained weights:
  Uses official torchvision ImageNet-1K pretrained weights (downloaded automatically
  by PyTorch when running with internet on Kaggle). No manual download needed.
"""

import os
import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet34, ResNet34_Weights,
    resnet18, ResNet18_Weights,
)


class ResNetBackbone(nn.Module):
    """
    4-stage multi-scale feature extractor based on torchvision ResNet.
    """
    def __init__(self, arch: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        self.arch = arch

        if arch == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            try:
                rn = resnet50(weights=weights)
                if pretrained:
                    print('[Backbone] ResNet-50 initialized with official ImageNet-1K pretrained weights.')
            except Exception as e:
                print(f'[Backbone] Warning: Could not download online weights ({e}). Initializing ResNet-50 randomly.')
                rn = resnet50(weights=None)
            self.num_features = [256, 512, 1024, 2048]

        elif arch == 'resnet34':
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            try:
                rn = resnet34(weights=weights)
                if pretrained:
                    print('[Backbone] ResNet-34 initialized with official ImageNet-1K pretrained weights.')
            except Exception as e:
                print(f'[Backbone] Warning: Could not download online weights ({e}). Initializing ResNet-34 randomly.')
                rn = resnet34(weights=None)
            self.num_features = [64, 128, 256, 512]

        elif arch == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            try:
                rn = resnet18(weights=weights)
                if pretrained:
                    print('[Backbone] ResNet-18 initialized with official ImageNet-1K pretrained weights.')
            except Exception as e:
                print(f'[Backbone] Warning: Could not download online weights ({e}). Initializing ResNet-18 randomly.')
                rn = resnet18(weights=None)
            self.num_features = [64, 128, 256, 512]

        else:
            raise NotImplementedError(f"Unsupported ResNet architecture: {arch}")

        # Stem: conv1 (7x7, s=2) -> bn1 -> relu -> maxpool (3x3, s=2)
        # Input (B, 3, H, W) -> Stem Output (B, 64, H/4, W/4)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)

        # 4 Residual Stages:
        # layer1: stride 1 -> (B, C1, H/4,  W/4)
        # layer2: stride 2 -> (B, C2, H/8,  W/8)
        # layer3: stride 2 -> (B, C3, H/16, W/16)
        # layer4: stride 2 -> (B, C4, H/32, W/32)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            list of 4 feature maps [f1, f2, f3, f4], all in standard (B, C, H, W) format.
        """
        x = self.stem(x)
        f1 = self.layer1(x)    # Stage 1: (B, C1, H/4,  W/4)
        f2 = self.layer2(f1)   # Stage 2: (B, C2, H/8,  W/8)
        f3 = self.layer3(f2)   # Stage 3: (B, C3, H/16, W/16)
        f4 = self.layer4(f3)   # Stage 4: (B, C4, H/32, W/32)
        return [f1, f2, f3, f4]


def build_resnet50_backbone(pretrained: bool = True) -> ResNetBackbone:
    return ResNetBackbone(arch='resnet50', pretrained=pretrained)


def build_resnet34_backbone(pretrained: bool = True) -> ResNetBackbone:
    return ResNetBackbone(arch='resnet34', pretrained=pretrained)


def build_resnet18_backbone(pretrained: bool = True) -> ResNetBackbone:
    return ResNetBackbone(arch='resnet18', pretrained=pretrained)
