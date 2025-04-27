import wandb
from lightning import Trainer

from data_tools.data_module import DataModule
from models import Model, MODELS
from settings import Config, ModelConfig, ModelName
from utils.ddp import is_main_proc
from utils.logger import logger
from utils.metrics import calculate_metrics_per_epoch
from utils.run_id import get_run_id, set_run_id


def train(model: Model, config: Config, use_wandb: bool = True, use_pretraining: bool = False) -> None:
    trainer_params = config.get_trainer_params(use_wandb)
    trainer = Trainer(**trainer_params)
    logger.info('Creating data module')
    data_config = config.pre_data if use_pretraining else config.data
    data_module = DataModule(data_config, config.pre_process)

    logger.info('Training')
    trainer.fit(model, datamodule=data_module)
    logger.info('Training completed. Aggregating validation metrics')
    log_metrics(trainer, config.model.threshold)
    logger.info('Done aggregating validation metrics')


def test(config: Config, test_data_folder: str) -> None:
    model = get_model_from_checkpoint(config)
    data_module = DataModule(config.data, config.pre_process, test_data_folder)

    test_params = config.get_tester_params()
    tester = Trainer(**test_params)
    logger.info(f'Testing on {test_data_folder}')
    tester.test(model=model, datamodule=data_module)
    logger.info('Done testing')


def load_model(checkpoint_path: str, model_name: ModelName, model_config: ModelConfig):
    model_class = MODELS[model_name]
    logger.info(f'Loading model {model_name.value} from {checkpoint_path}')
    return model_class.load_from_checkpoint(str(checkpoint_path), config=model_config)


def log_metrics(trainer: Trainer, threshold: float) -> None:
    if is_main_proc():
        metrics = calculate_metrics_per_epoch(get_run_id(), threshold)
        for metric_name, values in metrics.items():
            for epoch, value in enumerate(values):
                trainer.logger.experiment.log({metric_name: value, "epoch": epoch})


def restart_wandb_run(config: Config, run_postfix: str) -> None:
    if is_main_proc():
        run_name = f'{config.get_checkpoint_name()}_{run_postfix}'
        wandb.finish()
        wandb.init(name=run_name, reinit=True, config=config.get_wandb_params())
        wandb.run.log_code(".")
        set_run_id(wandb.run.id)


def get_model_from_checkpoint(config: Config) -> Model:
    model_class = MODELS[config.model_name]
    checkpoint_path = config.model_checkpoint_cb.best_model_path
    logger.info(f'Loading model {config.model_name.value} from {checkpoint_path}')
    return model_class.load_from_checkpoint(str(checkpoint_path), config=config.model)
