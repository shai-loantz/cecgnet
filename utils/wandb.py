import argparse
from ast import literal_eval

import wandb

from settings import Config
from utils.ddp import is_main_proc
from utils.logger import LOG_DIR
from utils.run_id import set_run_id


def start_wandb_sweep(config: Config, run_postfix: str) -> Config:
    if is_main_proc():
        wandb_config = parse_wandb_sweep()
        config.update_wandb_config(wandb_config)
        run_name = f'{config.get_checkpoint_name()}_{run_postfix}'
        wandb.init(name=run_name, resume=True)
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
