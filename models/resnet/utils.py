from enum import Enum

import torch
from torch import nn

from models.blocks import BasicConv

BASE_CHANNELS = 64
CHANNEL_EXPANSION = 4
HEAD_HIDDEN_DIM: tuple[int] = tuple()


class LayerConf(Enum):
    SMALL = (2, 2, 2, 2)  # Like resnet_18 but with bottleneck (26 layers)
    RESNET_50 = (3, 4, 6, 3)
    RESNET_101 = (3, 4, 23, 3)
    RESNET_152 = (3, 8, 36, 3)


class ResNetBlock(nn.Module):
    """ResNet block with bottleneck"""

    def __init__(self, in_channels: int, base_channels: int, expansion: int = 4, downsample: bool = False) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = BasicConv(in_channels, base_channels, kernel_size=1)
        self.conv2 = BasicConv(base_channels, base_channels, kernel_size=3, padding=1, stride=stride)
        self.conv3 = BasicConv(base_channels, base_channels * expansion, activation=False, kernel_size=1)

        self.identity_downsample = BasicConv(in_channels,
                                             base_channels * expansion,
                                             activation=False,
                                             kernel_size=1,
                                             stride=2,
                                             bias=False) if downsample else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_downsample(x) if self.identity_downsample else x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.activation(x + identity)


def get_tail_module(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        BasicConv(in_channels, BASE_CHANNELS, kernel_size=7, stride=2, padding=3, bias=False),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    )


def get_resnet_blocks(layer_conf: LayerConf, base_channels: int = BASE_CHANNELS) -> nn.ModuleList:
    blocks = nn.ModuleList()
    in_channels = base_channels
    for layer_size in layer_conf.value:
        blocks.append(ResNetBlock(in_channels, base_channels, downsample=True))

        in_channels = base_channels * CHANNEL_EXPANSION
        for _ in range(layer_size - 1):
            blocks.append(ResNetBlock(in_channels, base_channels))

        base_channels = base_channels * 2

    return blocks


def get_head_module(layer_conf: LayerConf) -> nn.Sequential:
    head_steps: list[nn.Module] = []
    in_features = BASE_CHANNELS * (2 ** (len(layer_conf.value) + 1))  # number of channels after last conv
    for out_features in HEAD_HIDDEN_DIM:
        head_steps.append(nn.Linear(in_features, out_features))
        head_steps.append(nn.ReLU())
        in_features = out_features
    head_steps.append(nn.Linear(in_features, 1))
    return nn.Sequential(*head_steps)
