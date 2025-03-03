import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from helper_code import find_records, load_label, load_signals


class ECGDataset(Dataset):
    def __init__(self, data_folder: str):
        self.record_files = [os.path.join(data_folder, record) for record in find_records(data_folder)]
        if not self.record_files:
            raise FileNotFoundError('No data were provided.')

    def __len__(self):
        return len(self.record_files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        record_file_name = self.record_files[idx]
        features = self.extract_features(record_file_name)
        labels = torch.tensor([load_label(record_file_name)], dtype=torch.float32)
        return features, labels

    @staticmethod
    def extract_features(record_file_name: str, training: bool = True):
        signal, fields = load_signals(record_file_name)

        # make sure leads are in the right order
        lead_order = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        index_map = [fields['sig_name'].index(lead) for lead in lead_order]
        signal = signal[:, index_map]
        # pad to 4096. TODO: make this generic
        pad_width = ((0, 4096 - signal.shape[0]), (0, 0))
        signal = np.pad(signal, pad_width)
        # TODO: pre-process
        input_type = torch.bfloat16 if training else torch.float
        return torch.tensor(signal, dtype=input_type).T
