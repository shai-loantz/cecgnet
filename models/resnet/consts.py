from enum import Enum

BASE_CHANNELS = 64
CHANNEL_EXPANSION = 4
HEAD_HIDDEN_DIM: tuple[int] = tuple()


class LayerConf(Enum):
    SMALL = (2, 2, 2, 2)  # Like resnet_18 but with bottleneck (26 layers)
    RESNET_50 = (3, 4, 6, 3)
    RESNET_101 = (3, 4, 23, 3)
    RESNET_152 = (3, 8, 36, 3)
