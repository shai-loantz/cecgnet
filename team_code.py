#!/usr/bin/env python
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from data_tools.data_loader import get_data_loaders
from data_tools.data_set import ECGDataset
from models import Model, MODELS
from settings import Config

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

config = Config()


def train_model(data_folder: str, model_folder: str, verbose: bool):
    if verbose:
        print('Finding the Challenge data...')
    train_loader, val_loader = get_data_loaders(data_folder, config)

    if verbose:
        print('Training the model on the data...')

    model = MODELS.get(config.model_name)(config.model)
    params = config.get_trainer_params()
    params['callbacks'] = [ModelCheckpoint(
        dirpath=model_folder,
        filename=config.checkpoint_name,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )]
    trainer = Trainer(**params)
    trainer.fit(model, train_loader, val_loader)

    if verbose:
        print('Done training')


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_class = MODELS.get(config.model_name)
    checkpoint_path = Path(model_folder) / f'{config.checkpoint_name}.ckpt'
    if verbose:
        print(f'Loading model {config.model_name} from {checkpoint_path}')
    return model_class.load_from_checkpoint(str(checkpoint_path), config=config.model)


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model: Model, verbose):
    model.eval()

    features = ECGDataset.extract_features(record)
    with torch.no_grad():
        output = model(features).detach().squeeze()
    probability_output = torch.sigmoid(output)
    binary_output = (probability_output > config.threshold).int()

    return binary_output, probability_output
