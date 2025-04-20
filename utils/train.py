from lightning import Trainer

from data_tools.data_module import DataModule
from models import Model
from settings import Config
from utils.ddp import is_main_proc
from utils.logger import logger
from utils.metrics import calculate_metrics_per_epoch
from utils.run_id import get_run_id


def run_train(model: Model, config: Config, use_wandb: bool = True, use_pretraining: bool = False):
    params = config.get_trainer_params(use_wandb)
    trainer = Trainer(**params)
    logger.info('Creating data module')
    data_config = config.pre_data if use_pretraining else config.data
    data_module = DataModule(data_config, config.pre_process)
    logger.info('Training')
    trainer.fit(model, datamodule=data_module)
    logger.info('Training completed. Evaluating')
    log_metrics(trainer, config.model.threshold)
    logger.info('Done Evaluating')


def log_metrics(trainer: Trainer, threshold: float) -> None:
    if is_main_proc():
        metrics = calculate_metrics_per_epoch(get_run_id(), threshold)
        for metric_name, values in metrics.items():
            for epoch, value in enumerate(values):
                trainer.logger.experiment.log({metric_name: value, "epoch": epoch})
