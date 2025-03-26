from lightning import Trainer

from data_tools.data_loader import get_data_loaders
from models import Model
from settings import DataLoaderConfig, PreprocessConfig


def run_train(verbose: bool, model: Model, trainer_params: dict, preprocess: PreprocessConfig,
              loader: DataLoaderConfig) -> Model:
    if verbose:
        print('finding training data...')
    train_loader, val_loader = get_data_loaders(loader, preprocess)
    if verbose:
        print('training model...')
    trainer = Trainer(**trainer_params)
    trainer.fit(model, train_loader, val_loader)
    return model
