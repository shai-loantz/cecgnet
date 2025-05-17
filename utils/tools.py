import argparse
from ast import literal_eval

import wandb
from lightning import Trainer
from wandb import Settings

from data_tools.data_module import DataModule
from models import Model, MODELS
from settings import Config, ModelConfig, ModelName, AugmentationsConfig
from utils.ddp import is_main_proc
from utils.logger import logger, LOG_DIR
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
    # logger.info('Training completed. Aggregating validation metrics')
    # log_metrics(trainer, config.model.threshold)
    logger.info('Done aggregating validation metrics')


def test(config: Config) -> None:
    model = get_model_from_checkpoint(config)
    data_module = DataModule(config.data, config.pre_process)

    test_params = config.get_tester_params()
    tester = Trainer(**test_params)
    logger.info(f'Testing')
    tester.test(model=model, datamodule=data_module)
    logger.info('Done testing')


def load_model(checkpoint_path: str, model_name: ModelName, model_config: ModelConfig, augmentations: AugmentationsConfig):
    model_class = MODELS[model_name]
    logger.info(f'Loading model {model_name.value} from {checkpoint_path}')
    return model_class.load_from_checkpoint(str(checkpoint_path), config=model_config, augmentations=augmentations)


def log_metrics(trainer: Trainer, threshold: float) -> None:
    if is_main_proc():
        metrics = calculate_metrics_per_epoch(get_run_id(), threshold)
        for metric_name, values in metrics.items():
            for epoch, value in enumerate(values):
                trainer.logger.experiment.log({metric_name: value, "epoch": epoch})


def start_wandb_sweep(config: Config, run_postfix: str) -> Config:
    if is_main_proc():
        wandb_config = parse_wandb_sweep()
        config.update_wandb_config(wandb_config)
        run_name = f'{config.get_checkpoint_name()}_{run_postfix}'
        # right now the sweep doesn't work with logs transferred to wandb correctly, from what i gather this happens
        # because of logger still writing things after the run finishes and wandb starts an endless loop
        wandb.init(settings=Settings(console="off"), name=run_name, resume=True)
        for key, value in config.get_wandb_params().items():
            wandb.config.update({key: value}, allow_val_change=False)
        wandb.config.update({"logs_dir": LOG_DIR})
        wandb.run.log_code(".")
        set_run_id(wandb.run.id)
        # please notice that sweep will only work with a single run and not pretraining + fine tuning
    return config


def restart_wandb_run(config: Config, run_postfix: str) -> None:
    if is_main_proc():
        run_name = f'{config.get_checkpoint_name()}_{run_postfix}'
        wandb.finish()
        wandb.init(project='cecgnet', name=run_name, reinit=True, config=config.get_wandb_params())
        wandb.run.log_code(".")
        set_run_id(wandb.run.id)


def get_model_from_checkpoint(config: Config) -> Model:
    # TODO: Make this work with a proper entry for the challenge (when we don't pre-train)
    model_class = MODELS[config.model_name]
    checkpoint_path = config.model_checkpoint_cb.best_model_path
    if checkpoint_path == '':
        raise Exception('No checkpoint was saved')
    logger.info(f'Loading model {config.model_name.value} from {checkpoint_path}')
    return model_class.load_from_checkpoint(str(checkpoint_path), config=config.model, augmentations=config.augmentations)


def parse_wandb_sweep() -> dict:
    parser = argparse.ArgumentParser()
    _, unknown = parser.parse_known_args()
    dynamic_args = {}
    for arg in unknown:
        if arg.startswith("--"):
            # Format: --key=value or --key value
            if '=' in arg:
                key, raw_value = arg.lstrip('-').split('=', 1)
                value = literal_eval(raw_value)
            else:
                key = arg.lstrip('-')
                value = True  # flag without value
            dynamic_args[key] = value

    return dynamic_args
