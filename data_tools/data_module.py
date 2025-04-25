import random

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader, Dataset

from data_tools.data_set import ECGDataset
from settings import DataConfig, PreprocessConfig
from utils.logger import logger


class DataModule(LightningDataModule):
    def __init__(self, data_config: DataConfig, preprocess_config: PreprocessConfig, test_data_folder: str | None = None) -> None:
        super().__init__()
        self.data_config = data_config
        self.preprocess_config = preprocess_config
        self.data_loader_config = self.data_config.get_data_loader_config()
        self.test_data_folder = test_data_folder

    def setup(self, stage: str) -> None:
        if stage == "fit":
            data_set = ECGDataset(self.data_config.data_folder, self.data_config.input_length, self.preprocess_config)
            length = len(data_set)
            valid_size = int(length * self.data_config.validation_size)
            train_size = length - valid_size
            logger.info(f'Train set size: {train_size}, Validation set size: {valid_size}')
            self.train_dataset, self.val_dataset = random_split(data_set, [train_size, valid_size])
        if stage == "test":
            self.test_dataset = ECGDataset(self.test_data_folder, self.data_config.input_length, self.preprocess_config)

    def train_dataloader(self):
        return self._get_data_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_data_loader(self.val_dataset)

    def test_dataloader(self):
        return self._get_data_loader(self.test_dataset, num_workers=0, persistent_workers=False, prefetch_factor=None)

    def _get_data_loader(self, data_set: Dataset, shuffle: bool = False, **kwargs) -> DataLoader:
        data_loader_config = self.data_loader_config.copy()
        data_loader_config.update(kwargs)
        return DataLoader(data_set, shuffle=shuffle,
                          worker_init_fn=seed_worker, **data_loader_config)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
