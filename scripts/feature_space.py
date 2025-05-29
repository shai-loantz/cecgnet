import numpy as np
import torch
from torch import Tensor, tensor

from models import Model, SimpleResNet
from scripts.input_space import get_inputs, reduce, plot
from settings import Config

CHECKPOINT_PATH = '/home/stu2/cecgnet/checkpoints/pretraining_resnet-v8.ckpt'

config = Config()


def get_model_from_checkpoint() -> Model:
    print(f'Loading model {config.model_name.value} from {CHECKPOINT_PATH}')
    return SimpleResNet.load_from_checkpoint(str(CHECKPOINT_PATH), config=config.model)


# Specific to ResNet!!!
def cnn_forward(model, x: Tensor) -> np.ndarray:
    x = x.to(next(model.parameters()).device)
    x = model.tail(x)
    for block in model.resnet_blocks:
        x = block(x)
    feature_space = model.gap(x).squeeze(-1)
    return feature_space.cpu().detach().numpy()


def main() -> None:
    x, dataset_labels = get_inputs()
    print(f'{x.shape=}')

    x_flat = x.reshape(x.shape[0], -1)
    print(f'{x.shape=}, {x_flat.shape=}')
    input_embeddings = reduce(x_flat)
    print(f'{input_embeddings.shape=}')
    plot(input_embeddings, dataset_labels, 'input_space_3d_datasets')

    model = get_model_from_checkpoint()
    feature_space = cnn_forward(model, tensor(x, dtype=torch.float))
    print(f'{feature_space.shape=}')
    feature_embeddings = reduce(feature_space)
    plot(feature_embeddings, dataset_labels, 'feature_space_3d_datasets')


if __name__ == "__main__":
    main()
