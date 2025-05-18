from torch import Tensor, nn

from models.model import Model
from models.resnet.utils import get_head_module, LayerConf
from models.resnet_attention.utils import get_attention_layers, get_attention_tail
from settings import ModelConfig, AugmentationsConfig


class ResNetAttention(Model):
    layer_conf = LayerConf.RESNET_101

    def __init__(self, config: ModelConfig, augmentations: AugmentationsConfig) -> None:
        super().__init__(config, augmentations)
        self.tail = get_attention_tail(config.input_channels, config.attention)
        self.layers = get_attention_layers(self.layer_conf, config.attention)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = get_head_module(self.layer_conf, config.add_metadata_end)

    def forward(self, x: Tensor, metadata: Tensor = None) -> Tensor:
        x = self.tail(x)
        for layer in self.layers:
            x = layer(x)
        x = self.gap(x).squeeze(-1)
        x = self.add_metadata(x, metadata) if self.config.add_metadata_end else x
        return self.head(x)


if __name__ == "__main__":
    ResNetAttention.test_model()
