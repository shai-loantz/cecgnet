from torch import nn, Tensor

from models import Model
from settings import ModelConfig, AugmentationsConfig


class VGG1D(Model):
    def __init__(self, config: ModelConfig, augmentations: AugmentationsConfig):
        super(VGG1D, self).__init__(config, augmentations)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.input_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        fc_in_features = 14848
        if config.add_metadata_end: fc_in_features += 2
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_in_features, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 1),
        )

    def forward(self, x: Tensor, metadata: Tensor = None) -> Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.add_metadata(x, metadata) if self.config.add_metadata_end else x
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    VGG1D.test_model()
