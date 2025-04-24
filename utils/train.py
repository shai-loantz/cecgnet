import wandb
from lightning import Trainer

from data_tools.data_module import DataModule
from models import Model
from settings import Config
from utils.ddp import is_main_proc
from utils.logger import logger
from utils.metrics import calculate_metrics_per_epoch
from utils.run_id import get_run_id, set_run_id


def train_and_evaluate(model: Model, config: Config, use_wandb: bool = True,
                       use_pretraining: bool = False, test_data_folder: str | None = None) -> None:
    params = config.get_trainer_params(use_wandb)
    trainer = Trainer(**params)
    logger.info('Creating data module')
    data_config = config.pre_data if use_pretraining else config.data
    data_module = DataModule(data_config, config.pre_process, test_data_folder)

    logger.info('Training')
    trainer.fit(model, datamodule=data_module)
    logger.info('Training completed. Aggregating validation metrics')
    log_metrics(trainer, config.model.threshold, 'val')
    logger.info('Done aggregating validation metrics')

    if test_data_folder is not None:
        logger.info(f'Testing on {test_data_folder}')
        trainer.test(datamodule=data_module)
        logger.info('Done testing. Aggregating metrics')
        log_metrics(trainer, config.model.threshold, 'test')
        logger.info('Done aggregating test metrics')


def log_metrics(trainer: Trainer, threshold: float, step_name: str) -> None:
    if is_main_proc():
        metrics = calculate_metrics_per_epoch(get_run_id(), threshold, step_name)
        for metric_name, values in metrics.items():
            for epoch, value in enumerate(values):
                trainer.logger.experiment.log({metric_name: value, "epoch": epoch})


def restart_wandb_run(checkpoint_name: str, run_postfix: str) -> None:
    if is_main_proc():
        run_name = f'{checkpoint_name}_{run_postfix}'
        wandb.finish()
        wandb.init(name=run_name, reinit=True)
        set_run_id(wandb.run.id)
