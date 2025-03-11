from typing import Type

from models.model import Model
from models.resnet.model import SimpleResNet
from models.resnet_attention.model import ResNetAttention

MODELS: dict[str, Type[Model]] = {
    'simple': SimpleResNet,
    'resnet_attention': ResNetAttention,
}
