import argparse

from helper_code import *
from models import MODELS
from settings import Config
from utils import run_train

config = Config()

def get_parser():
    description = 'Train the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str)
    parser.add_argument('-m', '--model_folder', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

def train_model(model_folder: str, data_folder: str, verbose: bool):
    if data_folder:
        # assumes that if you gave parameters through text you got both
        config.update_settings(data_folder, model_folder)

    if config.pretraining:
        if verbose:
            print('Pre Training the model...')
        model = MODELS.get(config.model_name)(config.pre_model)
        params = config.get_trainer_params()
        run_train(verbose, model, params, config.pre_process, config.pre_loader)

        model.change_params(config.model)  # also saves the pretraining
        config.pretraining = False
        if verbose:
            print('Pre Training completed.')
    else:  # load from pretrained model
        model = load_model(config.pretraining_checkpoint_path, verbose)
    if verbose:
        print('Training the model...')
    params = config.get_trainer_params()
    run_train(verbose, model, params, config.pre_process, config.data_loader)
    if verbose:
        print('Training completed.')
    return model


def load_model(checkpoint_path: str, verbose: bool):
    model_class = MODELS.get(config.model_name)
    if verbose:
        print(f'Loading model {config.model_name.value} from {checkpoint_path}')
    return model_class.load_from_checkpoint(str(checkpoint_path), config=config.model)


if __name__ == '__main__':
    args = get_parser().parse_args(sys.argv[1:])
    train_model(args.model_folder, args.data_folder, args.verbose)
