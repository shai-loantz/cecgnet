import secrets

from lightning import seed_everything

from models import MODELS
from settings import Config
from utils.logger import logger
from utils.train import train, restart_wandb_run, load_model, test

seed_everything(42)
config = Config()
RUN_POSTFIX = secrets.token_hex(2)


def main():
    restart_wandb_run(config, RUN_POSTFIX)
    if config.pretraining:
        logger.info('Pre-training')
        model = MODELS.get(config.model_name)(config.pre_model)
        train(model, config, use_pretraining=True)

        model.change_params(config.model)  # also saves the pretraining
        config.pretraining = False
        restart_wandb_run(config, RUN_POSTFIX)
        logger.info('Pre-training completed')
    else:  # load from pretrained model
        model = load_model(config.pretraining_checkpoint_path,
                           config.model_name, config.model)
    logger.info('Fine-tuning')
    train(model, config)
    test(config)
    return model


if __name__ == '__main__':
    import wandb

    try:
        main()
    finally:
        wandb.finish()
