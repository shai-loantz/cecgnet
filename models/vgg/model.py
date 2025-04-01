from torch import nn

from models import Model
from settings import ModelConfig


class VGG1D(Model):
    def __init__(self, config: ModelConfig):
        super(VGG1D, self).__init__(config)
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

        self.fc_layers = nn.Sequential(
            nn.Linear(14848, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    VGG1D.test_model()
