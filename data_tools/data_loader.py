import torch.distributed as dist
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_tools.data_set import ECGDataset
from settings import DataLoaderConfig, PreprocessConfig
from utils.ddp import is_main_proc
from utils.logger import logger


def create_data_loaders(train_dataset: ECGDataset, val_dataset: ECGDataset,
                        data_loader_config: dict) -> tuple[DataLoader, DataLoader | None]:
    train_sampler = _get_train_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, **data_loader_config)
    if not is_main_proc:
        logger.info('Validation would not run in this rank')
        val_loader = None
    else:
        logger.info('Creating validation loader')
        val_loader = DataLoader(val_dataset, shuffle=False, **data_loader_config)
    logger.info(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    return train_loader, val_loader


def _get_train_sampler(train_dataset: ECGDataset) -> DistributedSampler | None:
    if not dist.is_initialized():
        return None
    return DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                              rank=dist.get_rank(), shuffle=True)


def get_data_loaders(config: DataLoaderConfig, preprocess: PreprocessConfig) -> tuple[DataLoader, DataLoader | None]:
    data_set = ECGDataset(config.data_folder, config.input_length, preprocess)
    length = len(data_set)
    valid_size = int(length * config.validation_size)
    train_size = length - valid_size

    train_dataset, val_dataset = random_split(data_set, [train_size, valid_size])
    return create_data_loaders(train_dataset, val_dataset, config.get_data_loader_config())
