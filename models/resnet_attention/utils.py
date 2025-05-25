from torch import nn

from models.blocks import ATTENTION
from models.resnet.consts import BASE_CHANNELS, CHANNEL_EXPANSION, LayerConf
from models.resnet.utils import ResNetBlock, get_tail_module
from settings import Attention


def get_attention_layers(layer_conf: LayerConf, attention_type: Attention,
                         base_channels: int = BASE_CHANNELS) -> nn.ModuleList:
    attention_class = ATTENTION[attention_type]
    layers = nn.ModuleList()
    in_channels = base_channels
    for layer_size in layer_conf.value:
        layer = nn.ModuleList()
        layer.append(ResNetBlock(in_channels, base_channels, downsample=True))

        in_channels = base_channels * CHANNEL_EXPANSION
        for _ in range(layer_size - 1):
            layer.append(ResNetBlock(in_channels, base_channels))

        layer.append(attention_class(in_channels=in_channels))
        layers.append(nn.Sequential(*layer))

        base_channels = base_channels * 2

    return layers


def get_attention_tail(input_channels: int, attention_type: Attention) -> nn.Sequential:
    attention_class = ATTENTION[attention_type]
    tail = get_tail_module(input_channels)
    return nn.Sequential(attention_class(in_channels=input_channels),
                         tail,
                         attention_class(in_channels=BASE_CHANNELS))
