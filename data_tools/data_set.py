import os

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data_tools.preprocess import preprocess
from helper_code import find_records, load_label, load_signals
from settings import PreprocessConfig
from utils.logger import setup_logger


class ECGDataset(Dataset):
    def __init__(self, data_folder: str, input_length: int, preprocess_config: PreprocessConfig) -> None:
        self.input_length = input_length
        self.record_files = [os.path.join(data_folder, record) for record in find_records(data_folder)]
        self.preprocess_config = preprocess_config
        if not self.record_files:
            raise FileNotFoundError('No data was provided.')
        self.logger = setup_logger()

    def __len__(self):
        return len(self.record_files)

    def get_label(self, idx: int) -> Tensor:
        record_file_name = self.record_files[idx]
        return torch.tensor([load_label(record_file_name)], dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        record_file_name = self.record_files[idx]
        try:
            features = extract_features(record_file_name, self.input_length, self.preprocess_config)
        except Exception:
            self.logger.exception(f'Failed extracting features for {record_file_name} ({idx=})')
            features = torch.zeros(12, 934, dtype=torch.bfloat16)
        label = torch.tensor([load_label(record_file_name)], dtype=torch.float32)
        return features, label


def extract_features(record_file_name: str, input_length: int, config: PreprocessConfig, training: bool = True) -> Tensor:
    signal, fields = load_signals(record_file_name)
    signal = preprocess(signal, fields['sig_name'], fields['fs'], input_length, config)
    input_type = torch.bfloat16 if training else torch.float
    return torch.tensor(signal, dtype=input_type).T
