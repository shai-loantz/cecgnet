import json

import numpy as np
from pecg.Preprocessing import Preprocessing
from tqdm import tqdm

from data_tools.data_set import ECGDataset
from data_tools.preprocess import preprocess
from helper_code import load_signals
from settings import Config


def is_ecg_acceptable(signal: np.ndarray, fs: float,
                      good_threshold=0.8,
                      min_good_leads=8,
                      bad_lead_threshold=0.2,
                      max_bad_leads=1):
    pre = Preprocessing(signal, int(fs))
    bsqi_per_lead = pre.bsqi()
    good_leads = bsqi_per_lead > good_threshold
    bad_leads = bsqi_per_lead < bad_lead_threshold

    return (np.sum(good_leads) >= min_good_leads) and (np.sum(bad_leads) <= max_bad_leads)


def main():
    print('Setting up')
    bad_signal_ids = []
    config = Config()
    dataset = ECGDataset(config.data.data_folder, config.data.input_length, config.pre_process)
    print('Iterating')
    for idx in tqdm(dataset.record_files):
        record_file_name = dataset.record_files[idx]
        signal, fields = load_signals(record_file_name)
        signal = preprocess(signal, fields['sig_name'], fields['fs'],
                            config.pre_process.input_length, config.pre_process)
        if not is_ecg_acceptable(signal, config.pre_process.resample_freq):
            bad_signal_ids.append(idx)

    print(f'Done. Found {len(bad_signal_ids)} bad signals. Writing to bad_signal_ids.txt')
    with open('bad_signal_ids.txt', 'w') as fh:
        json.dump(bad_signal_ids, fh)


if __name__ == '__main__':
    main()
