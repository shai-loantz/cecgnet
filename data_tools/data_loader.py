from torch.utils.data import random_split, DataLoader

from data_tools.data_set import ECGDataset
from settings import Config


def create_data_loaders(train_dataset: ECGDataset,
                        val_dataset: ECGDataset,
                        data_loader_config: dict) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, shuffle=True, **data_loader_config)
    val_loader = DataLoader(val_dataset, shuffle=False, **data_loader_config)
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}')
    return train_loader, val_loader


def get_data_loaders(data_folder: str, config: Config) -> tuple[DataLoader, DataLoader]:
    data_set = ECGDataset(data_folder)
    length = len(data_set)
    valid_size = int(length * config.validation_size)
    train_size = length - valid_size

    train_dataset, val_dataset = random_split(data_set, [train_size, valid_size])
    return create_data_loaders(train_dataset, val_dataset, config.data_loader.model_dump())
