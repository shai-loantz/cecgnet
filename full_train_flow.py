import secrets

from lightning import seed_everything

from models import MODELS
from settings import Config
from utils.logger import logger
from utils.train import train_and_evaluate, restart_wandb_run

seed_everything(42)
config = Config()
RUN_POSTFIX = secrets.token_hex(2)


def main():
    restart_wandb_run(config.get_checkpoint_name(), RUN_POSTFIX)
    if config.pretraining:
        logger.info('Pre-training')
        model = MODELS.get(config.model_name)(config.pre_model)
        train_and_evaluate(model, config, use_pretraining=True)

        model.change_params(config.model)  # also saves the pretraining
        config.pretraining = False
        restart_wandb_run(config.get_checkpoint_name(), RUN_POSTFIX)
        logger.info('Pre-training completed')
    else:  # load from pretrained model
        model = load_model(config.pretraining_checkpoint_path)
    logger.info('Fine-tuning')
    train_and_evaluate(model, config)
    return model


def load_model(checkpoint_path: str):
    model_class = MODELS[config.model_name]
    logger.info(f'Loading model {config.model_name.value} from {checkpoint_path}')
    return model_class.load_from_checkpoint(str(checkpoint_path), config=config.model)


if __name__ == '__main__':
    import wandb

    try:
        main()
    finally:
        wandb.finish()
