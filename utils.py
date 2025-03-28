from lightning import Trainer

from data_tools.data_loader import get_data_loaders
from models import Model
from settings import DataLoaderConfig, PreprocessConfig


def run_train(model: Model, trainer_params: dict, preprocess: PreprocessConfig,
              loader: DataLoaderConfig) -> None:
    print('finding training data...')
    train_loader, val_loader = get_data_loaders(loader, preprocess)
    print('training model...')
    trainer = Trainer(**trainer_params)
    trainer.fit(model, train_loader, val_loader)
