import torch
import random

from torchvision.transforms import v2

from settings import AugmentationsConfig


class ChannelErase:
    def __init__(self, channels, p=0.5, channel_to_drop=None):
        """
        Randomly zeroes out a selected channel in 1D multichannel input.

        Args:
            channels (int): Total number of channels in the input.
            p (float): Probability of applying the transform.
            channel_to_drop (int or None): If set, always drops this channel. If None, randomly selects one.
        """
        self.channels = channels
        self.p = p
        self.channel_to_drop = channel_to_drop

    def __call__(self, x): # TODO: do i need clone?
        if not torch.is_tensor(x):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[-2] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, but got {x.shape[-2]}")
        if x.dim() == 2:   # (C, L)
            if random.random() < self.p:
                c = self.channel_to_drop if self.channel_to_drop is not None else random.randint(0, self.channels - 1)
                x[c] = 0
        elif x.dim() == 3: # (B, C, L)
            for i in range(x.shape[0]):
                if random.random() < self.p:
                    c = self.channel_to_drop if self.channel_to_drop is not None else random.randint(0, self.channels - 1)
                    x[i, c] = 0
        else:
            raise ValueError(f"Expected shape (C, L) or (B, C, L), got {x.shape}")
        return x


def get_augmentations(config: AugmentationsConfig, input_channels=12):
    augmentation_options = (
        ("channel_erase", lambda: ChannelErase(input_channels)),
    )
    augmentations = [fn() for key, fn in augmentation_options if getattr(config, key)]
    if not [*augmentations]:
        return v2.Identity()
    return v2.RandomApply(augmentations, p=config.random_apply_probability)
