from typing import Type

from torch import nn

from models.blocks import SEBlock, SelfAttention, SequentialAttention
from models.model import Model
from models.resnet.model import SimpleResNet
from models.resnet_attention.model import ResNetAttention

MODELS: dict[str, Type[Model]] = {
    'simple': SimpleResNet,
    'resnet_attention': ResNetAttention,
}
ATTENTION: dict[str, Type[nn.Module]] = {
    'SE': SEBlock,
    'SelfAttention': SelfAttention,
    'SequentialAttention': SequentialAttention,
}
