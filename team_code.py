#!/usr/bin/env python
from pathlib import Path

import torch

from data_tools.data_set import extract_features
from models import Model, MODELS
from settings import Config
from utils import run_train

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

config = Config()


def train_model(data_folder: str, model_folder: str, verbose: bool):
    config.update_settings(data_folder, model_folder)
    model = MODELS.get(config.model_name)(config.model)
    params = config.get_trainer_params()
    run_train(verbose, model, params, config.pre_process, config.data_loader)
    if verbose:
        print('Done training')


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_class = MODELS.get(config.model_name)
    checkpoint_path = Path(model_folder) / f'{config.get_checkpoint_name()}.ckpt'
    if verbose:
        print(f'Loading model {config.model_name.value} from {checkpoint_path}')
    return model_class.load_from_checkpoint(str(checkpoint_path), config=config.model)


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record: str, model: Model, verbose: bool) -> tuple[float, float]:
    model.eval()

    if verbose:
        print(f'{record=}')
        print('Extracting features')
    features = extract_features(record, config.data_loader.input_length, config.pre_process, training=False)

    if verbose:
        print('Predicting')
    with torch.no_grad():
        logit = model(features.unsqueeze(0)).detach().squeeze()

    if verbose:
        print('Converting logit to probability and binary output')
    probability_output = torch.sigmoid(logit)
    binary_output = (probability_output > config.model.threshold).float()

    return binary_output.item(), probability_output.item()
