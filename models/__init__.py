from typing import Type

from models.hubert.model import Hubert
from models.model import Model
from models.resnet.model import SimpleResNet
from models.resnet_attention.model import ResNetAttention
from models.small.model import Small
from models.vgg.model import VGG1D

MODELS: dict[str, Type[Model]] = {
    'resnet': SimpleResNet,
    'resnet_attention': ResNetAttention,
    'vgg': VGG1D,
    'small': Small,
    'hubert': Hubert,
}
