from helper_code import *
from models import MODELS
from settings import Config
from train_model import get_parser
from utils import run_train

config = Config()


def train_model(args):
    verbose = args.verbose
    if args.data_folder:
        # assumes that if you gave parameters through text you got both
        config.update_settings(args.data_folder, args.model_folder)

    if config.pretraining:
        if verbose:
            print('Pre Training the model...')
        model = MODELS.get(config.model_name)(config.pre_model)
        params = config.get_trainer_params()
        run_train(verbose, model, params, config.pre_process, config.pre_loader)

        model.change_params(config.model)
        config.pretraining = False
        if verbose:
            print('Pre Training completed.')

    else:
        model = MODELS.get(config.model_name)(config.model)
    if verbose:
        print('Training the model...')
    params = config.get_trainer_params()
    run_train(verbose, model, params, config.pre_process, config.data_loader)
    if verbose:
        print('Training completed.')
    return model


if __name__ == '__main__':
    train_model(get_parser().parse_args(sys.argv[1:]))
