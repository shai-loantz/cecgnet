from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_tools.data_set import ECGDataset
from settings import DataConfig, PreprocessConfig
from utils.logger import logger


class DataModule(LightningDataModule):
    def __init__(self, data_config: DataConfig, preprocess_config: PreprocessConfig) -> None:
        super().__init__()
        data_set = ECGDataset(data_config.data_folder, data_config.input_length, preprocess_config)
        length = len(data_set)
        valid_size = int(length * data_config.validation_size)
        train_size = length - valid_size

        self.data_loader_config = data_config.get_data_loader_config()
        self.train_dataset, self.val_dataset = random_split(data_set, [train_size, valid_size])

    def train_dataloader(self):
        if self.trainer.strategy and hasattr(self.trainer.strategy, 'local_rank') and self.trainer.strategy.local_rank != -1:
            logger.info('Using distributed sampler for training')
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
            )
        else:
            logger.info('Not using distributed sampler for training')
            sampler = None  # No sampler if not distributed

        return DataLoader(self.train_dataset, shuffle=(sampler is None),
                          sampler=sampler, **self.data_loader_config)

    def val_dataloader(self):
        # Only use validation data on rank 0, return empty DataLoader for other ranks
        if self.trainer.local_rank == 0:
            return DataLoader(self.val_dataset, shuffle=False, **self.data_loader_config)
        else:
            return DataLoader([])
