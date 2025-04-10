from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

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
        logger.info(f'Train set size: {train_size}, Validation set size: {valid_size}')

        self.data_loader_config = data_config.get_data_loader_config()
        self.train_dataset, self.val_dataset = random_split(data_set, [train_size, valid_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.data_loader_config)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.data_loader_config)
