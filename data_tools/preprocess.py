import numpy as np

from scipy.signal import resample, butter, sosfiltfilt

RESAMPLE_FS = 128
LOW_CUT = 0.5
HIGH_CUT = 45


def preprocess(signal: np.ndarray, signal_names: list[str], fs: int, input_length: int) -> np.ndarray:
    signal = reorder_leads(signal, signal_names)
    signal = filter_signal(signal, fs)
    signal = resample_signal(signal, fs, RESAMPLE_FS)
    signal = cut_signal(signal, input_length)

    return signal


def cut_signal(signal: np.ndarray, input_length: int) -> np.ndarray:
    return signal[:input_length, :]


def resample_signal(original_signal: np.ndarray, original_fs: int, new_fs: int):
    num_samples = int(len(original_signal) * (new_fs / original_fs))
    return resample(original_signal, num_samples)


def filter_signal(signal: np.ndarray, fs: int) -> np.ndarray:
    # not needed as HIGH_CUT is lower than 50, 60
    # pre = Preprocessing(signal, fs)
    # # 50 Hz for european powerline, 60 Hz for USA
    # pre.notch(50)
    # pre.notch(60)
    # signal = pre.signal

    return apply_band_pass_filter(signal, fs, LOW_CUT, HIGH_CUT)


def reorder_leads(signal: np.ndarray, signal_names: list[str]) -> np.ndarray:
    lead_order = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    index_map = [signal_names.index(lead) for lead in lead_order]
    return signal[:, index_map]


def apply_band_pass_filter(signal: np.ndarray, fs: int, low_cut: float, high_cut: float, filter_order: int = 75) -> np.ndarray:
    nyquist_freq = 0.5 * fs
    low = low_cut / nyquist_freq
    high = high_cut / nyquist_freq
    if fs <= high_cut * 2:
        sos = butter(filter_order, low, btype="high", output='sos', analog=False)
    else:
        sos = butter(filter_order, [low, high], btype="band", output='sos', analog=False)

    if len(np.shape(signal)) == 2:
        [ecg_len, ecg_num] = np.shape(signal)
        filtered_signal = np.zeros([ecg_len, ecg_num])
        for i in np.arange(0, ecg_num):
            filtered_signal[:, i] = sosfiltfilt(sos, signal[:, i])
    elif len(np.shape(signal)) == 1:
        filtered_signal = sosfiltfilt(sos, signal)
    else:
        raise Exception('len(np.shape(signal)) must be 1 or 2')

    return filtered_signal
