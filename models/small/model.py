from torch import nn, Tensor

from models import Model
from settings import ModelConfig, AugmentationsConfig


class Small(Model):
    def __init__(self, config: ModelConfig, augmentations: AugmentationsConfig):
        super(Small, self).__init__(config, augmentations)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.input_channels, 64, kernel_size=3, padding=1), self.activation,
            nn.Conv1d(64, 64, kernel_size=3, padding=1), self.activation,
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1), self.activation,
            nn.Conv1d(128, 128, kernel_size=3, padding=1), self.activation,
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        fc_in_features = 29824
        if config.add_metadata_end: fc_in_features += 2
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_in_features, 1),
        )

    def forward(self, x: Tensor, metadata: Tensor = None) -> Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.add_metadata(x, metadata) if self.config.add_metadata_end else x
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    Small.test_model()
