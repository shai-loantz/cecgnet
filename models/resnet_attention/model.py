from torch import Tensor, nn

from models.blocks import SelfAttention
from models.model import Model
from models.resnet.utils import get_head_module, get_resnet_blocks, LayerConf, get_tail_module
from settings import ModelConfig


class ResNetAttention(Model):
    layer_conf = LayerConf.SMALL  # working only with small for now

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.tail = get_tail_module(config.input_channels)
        self.layers = get_resnet_blocks(self.layer_conf)
        self.attention_blocks = nn.ModuleList((SelfAttention(12),
                                               SelfAttention(64),
                                               SelfAttention(256),
                                               SelfAttention(512),
                                               SelfAttention(1024),
                                               SelfAttention(2048)))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = get_head_module(self.layer_conf)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention_blocks[0](x)
        x = self.tail(x)
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x = self.attention_blocks[1 + i // 2](x)
            x = layer(x)
        x = self.attention_blocks[5](x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)


if __name__ == "__main__":
    ResNetAttention.test_model()
