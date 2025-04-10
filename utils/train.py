from lightning import Trainer

from data_tools.data_module import DataModule
from models import Model
from settings import Config
from utils.logger import logger


def run_train(model: Model, config: Config, use_wandb: bool = True, use_pretraining: bool = False) -> Trainer:
    params = config.get_trainer_params(use_wandb)
    trainer = Trainer(**params)
    logger.info('Creating data module')
    data_config = config.pre_data if use_pretraining else config.data
    data_module = DataModule(data_config, config.pre_process)
    logger.info('Training')
    trainer.fit(model, datamodule=data_module)
    return trainer
