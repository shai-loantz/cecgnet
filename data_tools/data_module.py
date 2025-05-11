import random

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, Sampler

from data_tools.data_set import ECGDataset
from settings import DataConfig, PreprocessConfig
from utils.logger import logger


class DataModule(LightningDataModule):
    def __init__(self, data_config: DataConfig, preprocess_config: PreprocessConfig) -> None:
        super().__init__()
        self.data_config = data_config
        self.preprocess_config = preprocess_config
        self.data_loader_config = self.data_config.get_data_loader_config()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            logger.debug('Getting dataset')
            dataset = ECGDataset(self.data_config.data_folder, self.data_config.input_length, self.preprocess_config)
            logger.debug('Getting labels')
            labels = [dataset.get_label(idx) for idx in range(len(dataset))]
            logger.debug('Getting Split indices')
            train_indices, validation_indices = train_test_split(
                range(len(dataset)),
                test_size=self.data_config.validation_size,
                stratify=labels,
                random_state=42,
            )
            logger.debug('Creating dataset Subsets')
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, validation_indices)
            logger.debug('Creating train_labels')
            self.train_labels = [labels[i] for i in train_indices]
            logger.info(f'Train set size: {len(train_indices)}, Validation set size: {len(validation_indices)}')
        if stage == "test":
            self.test_dataset = ECGDataset(self.data_config.test_data_folder, self.data_config.input_length, self.preprocess_config)
            logger.info(f'Train set size: {len(self.test_dataset)}')

    def train_dataloader(self):
        sample_weights = torch.tensor([
            1.0 if label == 0 else self.data_config.positive_sampling_factor
            for label in self.train_labels
        ])
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return self._get_data_loader(self.train_dataset, sampler=sampler)

    def val_dataloader(self):
        return self._get_data_loader(self.val_dataset)

    def test_dataloader(self):
        return self._get_data_loader(self.test_dataset)

    def _get_data_loader(self, data_set: Dataset, shuffle: bool = False, sampler: Sampler = None, **kwargs) -> DataLoader:
        data_loader_config = self.data_loader_config.copy()
        data_loader_config.update(kwargs)
        return DataLoader(data_set, shuffle=shuffle and (sampler is None), sampler=sampler,
                          worker_init_fn=seed_worker, **data_loader_config)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
