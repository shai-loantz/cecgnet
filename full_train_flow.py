import secrets

from lightning import seed_everything
from torch import set_float32_matmul_precision

from models import MODELS
from settings import Config
from utils.logger import logger
from utils.tools import train, restart_wandb_run, load_model, test, get_model_from_checkpoint, start_wandb_sweep

set_float32_matmul_precision('high')
seed_everything(42)
RUN_POSTFIX = secrets.token_hex(2)


def main():
    config = Config()
    if config.manual_config:
        restart_wandb_run(config, RUN_POSTFIX)
    else:
        config = start_wandb_sweep(config, RUN_POSTFIX)
    if config.pretraining:
        logger.info('Pre-training')
        model = MODELS[config.model_name](config.pre_model, config.augmentations)
        train(model, config, use_pretraining=True)
        model = get_model_from_checkpoint(config)
        test(config)

        model.change_params(config.model)  # also saves the pretraining
        config.pretraining = False
        restart_wandb_run(config, RUN_POSTFIX)
        logger.info('Pre-training completed')
    else:  # load from pretrained model
        try:
            model = load_model(config.pretraining_checkpoint_path, config.model_name, config.model, config.augmentations)
        except:
            logger.debug('load pre-training failed')
            model = MODELS[config.model_name](config.model, config.augmentations)
    logger.info('Fine-tuning')
    train(model, config)
    test(config)
    return model


if __name__ == '__main__':
    import wandb

    try:
        main()
    except KeyboardInterrupt:
        logger.error('Interrupted by user. Closing')
    finally:
        wandb.finish()
