import torch
from torch import nn

from models.blocks import BasicConv
from models.model import METADATA_DIM
from models.resnet.consts import BASE_CHANNELS, LayerConf, CHANNEL_EXPANSION, HEAD_HIDDEN_DIM
from models.utils import get_head_module


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
                                             stride=2) if downsample else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_downsample(x) if self.identity_downsample else x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.activation(x + identity)


def get_tail_module(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        BasicConv(in_channels, BASE_CHANNELS, kernel_size=7, stride=2, padding=3),
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


def get_resnet_head_module(layer_conf: LayerConf, add_metadata_end: bool) -> nn.Sequential:
    in_features = BASE_CHANNELS * (2 ** (len(layer_conf.value) + 1))  # number of channels after last conv
    return get_head_module(in_features, add_metadata_end)
