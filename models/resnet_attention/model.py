from torch import Tensor, nn

from models.model import Model
from models.resnet.utils import get_head_module, LayerConf
from models.resnet_attention.utils import get_attention_layers, get_attention_tail
from settings import ModelConfig


class ResNetAttention(Model):
    layer_conf = LayerConf.SMALL

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.tail = get_attention_tail(config.input_channels)
        self.layers = get_attention_layers(self.layer_conf, 'SelfAttention')
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = get_head_module(self.layer_conf)

    def forward(self, x: Tensor) -> Tensor:
        x = self.tail(x)
        for layer in self.layers:
            x = layer(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)


if __name__ == "__main__":
    ResNetAttention.test_model()
