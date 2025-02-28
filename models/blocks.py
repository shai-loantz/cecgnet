import torch
from torch import nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block - Channel soft attention. Reweight the channels according to their means.
    """

    def __init__(self, in_channels: int, reduction=16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1)  # mean of each channel over time
        attn = self.fc2(self.activation(self.fc1(x_mean)))
        return x * torch.sigmoid(attn).unsqueeze(1)


class SequentialAttention(nn.Module):
    """
    Uses the same concept of Spatial Attention in CBAM but in 1D instead of 2D.
    Sequential attention mechanism to reweight sequential parts according to max and average across channels (dim=1).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        attn_input = torch.cat([max_pool, avg_pool], dim=1)  # stack max and average as two channels
        attn = self.conv(attn_input)

        return x * self.sigmoid(attn)


class SelfAttention(nn.Module):
    """
    Sequential self attention using query, key and value embeddings (one head).
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.key_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.query_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We use permute(0, 2, 1) as transpose while keeping batches
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        attn_scores = torch.bmm(query.permute(0, 2, 1), key) / (x.shape[1] ** 0.5)
        attn_weights = self.softmax(attn_scores)

        return torch.bmm(value, attn_weights.permute(0, 2, 1))


class BasicConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, activation: bool = True, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, *args, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU() if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.conv(x))
        return self.activation(out) if self.activation else out
