import json
import os
import sys
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from pecg.Preprocessing import Preprocessing
from tqdm import tqdm

from data_tools.data_set import ECGDataset
from data_tools.preprocess import preprocess
from helper_code import load_signals
from settings import Config

GOOD_THRESHOLD = 0.75
BAD_THRESHOLD = 0.2
MIN_GOOD_LEADS = 8
MAX_BAD_LEADS = 1
NUM_WORKERS = 32


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def is_ecg_acceptable(signal: np.ndarray, fs: float) -> bool:
    pre = Preprocessing(signal, int(fs))
    with HiddenPrints():
        bsqi_per_lead = pre.bsqi()
    good_leads = bsqi_per_lead > GOOD_THRESHOLD
    bad_leads = bsqi_per_lead < BAD_THRESHOLD
    return (np.sum(good_leads) >= MIN_GOOD_LEADS) and (np.sum(bad_leads) <= MAX_BAD_LEADS)


def get_mean_bsqi(record_file_name: str) -> float | None:
    config = Config()
    signal, fields = load_signals(record_file_name)
    signal = preprocess(signal, fields['sig_name'], fields['fs'],
                        config.data.input_length, config.pre_process)
    pre = Preprocessing(signal, config.pre_process.resample_freq)
    with HiddenPrints():
        bsqi_per_lead = pre.bsqi()
    return bsqi_per_lead.mean()


def process_record(record_file_name: str) -> str | None:
    config = Config()
    try:
        signal, fields = load_signals(record_file_name)
        signal = preprocess(signal, fields['sig_name'], fields['fs'],
                            config.data.input_length, config.pre_process)
        if not is_ecg_acceptable(signal, config.pre_process.resample_freq):
            return record_file_name
    except Exception:
        return record_file_name  # Consider failed records as bad
    return None


def plot_histogram():
    print('Setting up')
    config = Config()
    dataset = ECGDataset(config.data.data_folder, config.data.input_length, config.pre_process)
    record_files = dataset.record_files

    print(f'Iterating using {NUM_WORKERS}/{cpu_count()} workers')
    bsqi = []
    with Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(get_mean_bsqi, record_files), total=len(record_files)):
            if result is not None:
                bsqi.append(result)

    bsqi = np.array(bsqi)
    plt.hist(bsqi)
    plt.show()


def main():
    print('Setting up')
    config = Config()
    dataset = ECGDataset(config.data.data_folder, config.data.input_length, config.pre_process)
    record_files = dataset.record_files

    print(f'Iterating using {NUM_WORKERS}/{cpu_count()} workers')
    bad_signal_ids = []
    with Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_record, record_files), total=len(record_files)):
            if result is not None:
                bad_signal_ids.append(result)

    print(f'Done. Found {len(bad_signal_ids)} bad signals which are {len(bad_signal_ids) * 100 / len(record_files)}%. '
          f'Writing to bad_signal_ids.txt')
    with open('bad_signal_ids.txt', 'w') as fh:
        json.dump(bad_signal_ids, fh)


if __name__ == '__main__':
    main()
