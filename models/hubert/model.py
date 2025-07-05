from torch import Tensor, nn
from transformers import AutoModel

from models.model import Model
from models.utils import get_head_module
from settings import ModelConfig, AugmentationsConfig


class Hubert(Model):
    def __init__(self, config: ModelConfig, augmentations: AugmentationsConfig) -> None:
        super().__init__(config, augmentations)
        self.model = AutoModel.from_pretrained('Edoardo-BS/hubert-ecg-small',
                                               trust_remote_code=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = get_head_module(14, self.config.add_metadata_end, (1024, 256))

    def forward(self, x: Tensor, metadata: Tensor = None) -> Tensor:
        x = x[:, 1, :]  # Use lead II
        x = self.model(x).last_hidden_state
        x = self.gap(x).squeeze(-1)
        if self.config.add_metadata_end:
            x = self.add_metadata(x, metadata)
        return self.head(x)


if __name__ == "__main__":
    Hubert.test_model()
