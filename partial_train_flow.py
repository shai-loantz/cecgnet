import secrets

from lightning import seed_everything

from models import MODELS
from settings import Config
from utils.tools import train, test, start_wandb_sweep, restart_wandb_run

seed_everything(42)
RUN_POSTFIX = secrets.token_hex(2)


def main() -> None:
    """
    This is a partial and temporary flow to compare models.
    This flow consists of training on the "pre-training" data set and then testing on the "fine-tuning" database.
    """
    config = Config()
    if config.manual_config:
        restart_wandb_run(config, RUN_POSTFIX)
    else:
        config = start_wandb_sweep(config, RUN_POSTFIX)
    model = MODELS[config.model_name](config.pre_model)
    train(model, config, use_pretraining=False)
    test(config)


if __name__ == '__main__':
    import wandb

    try:
        main()
    finally:
        wandb.finish()
