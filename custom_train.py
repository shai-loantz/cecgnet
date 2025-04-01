import secrets

from models import MODELS
from settings import Config
from utils.ddp import is_main_proc
from utils.logger import logger
from utils.train import run_train

config = Config()
RUN_POSTFIX = secrets.token_hex(2)


def train_model():
    restart_wandb_run()
    if config.pretraining:
        logger.info('Pre-training')
        model = MODELS.get(config.model_name)(config.pre_model)
        params = config.get_trainer_params()
        run_train(model, params, config.pre_process, config.pre_loader)

        model.change_params(config.model)  # also saves the pretraining
        config.pretraining = False
        restart_wandb_run()
        logger.info('Pre-training completed')
    else:  # load from pretrained model
        model = load_model(config.pretraining_checkpoint_path)
    logger.info('Fine-tuning')
    params = config.get_trainer_params()
    run_train(model, params, config.pre_process, config.data_loader)
    logger.info('Training completed.')
    return model


def restart_wandb_run():
    if is_main_proc():
        run_name = f'{config.get_checkpoint_name()}_{RUN_POSTFIX}'
        wandb.finish()
        wandb.init(name=run_name, reinit=True)


def load_model(checkpoint_path: str):
    model_class = MODELS[config.model_name]
    logger.info(f'Loading model {config.model_name.value} from {checkpoint_path}')
    return model_class.load_from_checkpoint(str(checkpoint_path), config=config.model)


if __name__ == '__main__':
    import wandb

    try:
        train_model()
    finally:
        wandb.finish()
