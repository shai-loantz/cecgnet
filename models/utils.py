from torch import nn

from models.model import METADATA_DIM
from models.resnet.consts import HEAD_HIDDEN_DIM


def get_head_module(in_features: int, add_metadata_end: bool, head_hidden_dim = None) -> nn.Sequential:
    if head_hidden_dim is None:
        head_hidden_dim = HEAD_HIDDEN_DIM
    head_steps: list[nn.Module] = []
    if add_metadata_end: in_features += METADATA_DIM
    for out_features in head_hidden_dim:
        head_steps.append(nn.Linear(in_features, out_features))
        head_steps.append(nn.ReLU())
        in_features = out_features
    head_steps.append(nn.Linear(in_features, 1))
    return nn.Sequential(*head_steps)
