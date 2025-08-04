from torch import Tensor, nn

from models.model import Model
from models.resnet.utils import get_resnet_head_module, get_resnet_blocks, LayerConf, get_tail_module
from settings import ModelConfig, AugmentationsConfig


class SimpleResNet(Model):
    layer_conf = LayerConf.RESNET_101

    def __init__(self, config: ModelConfig, augmentations: AugmentationsConfig) -> None:
        super().__init__(config, augmentations)
        self.tail = get_tail_module(config.input_channels)
        self.resnet_blocks = get_resnet_blocks(self.layer_conf, self.activation)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = get_head_module(self.layer_conf, self.config.add_metadata_end, self.activation)

    def forward(self, x: Tensor, metadata: Tensor = None) -> Tensor:
        x = self.tail(x)
        for block in self.resnet_blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        x = self.add_metadata(x, metadata) if self.config.add_metadata_end else x
        return self.head(x)


if __name__ == "__main__":
    SimpleResNet.test_model()
