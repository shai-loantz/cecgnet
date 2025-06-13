import os

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data_tools.preprocess import preprocess
from helper_code import find_records, load_label, load_signals, get_age, load_header, get_sex
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

    def get_metadata(self, idx: int) -> Tensor:
        # TODO: change handling of missing data?
        record_file_name = self.record_files[idx]
        header = load_header(record_file_name)
        age = get_age(header)
        age = float(age) if age is not None else -1.0
        sex_str = get_sex(header)
        sex_map = {"Female": 1.0, "Male": -1.0}
        # returns the sex float in accordance with sex_map, if got None returns 0, if got unidentified str returns None
        sex = sex_map.get(sex_str, 0.0 if sex_str is None else None)
        if sex is None:
            raise NotImplementedError(f"Unrecognized sex: {sex_str}")
        return torch.tensor([age, sex], dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        record_file_name = self.record_files[idx]
        try:
            features = extract_features(record_file_name, self.input_length, self.preprocess_config)
        except Exception:
            self.logger.exception(f'Failed extracting features for {record_file_name} ({idx=})')
            features = torch.zeros(12, 934, dtype=torch.bfloat16)
        label = self.get_label(idx)
        metadata = self.get_metadata(idx)
        return features, label, metadata


def extract_features(record_file_name: str, input_length: int, config: PreprocessConfig,
                     training: bool = True) -> Tensor:
    signal, fields = load_signals(record_file_name)
    signal = preprocess(signal, fields['sig_name'], fields['fs'], input_length, config)
    input_type = torch.bfloat16 if training else torch.float
    return torch.tensor(signal, dtype=input_type).T
