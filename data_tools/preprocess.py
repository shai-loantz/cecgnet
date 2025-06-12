import numpy as np
import pywt
from scipy.signal import resample, butter, sosfiltfilt
import random

from settings import PreprocessConfig
from utils.logger import logger

POSSIBLE_FS = (400, 500)  # Sampling frequencies of our known datasets
FILTER_ORDER = 10
filters: dict = {}


def preprocess(signal: np.ndarray, signal_names: list[str], fs: int, input_length: int, config: PreprocessConfig) -> np.ndarray:
    signal = trim_zeros(signal)
    signal = reorder_leads(signal, signal_names)
    signal = np.apply_along_axis(remove_baseline_wander, 0, signal)
    sos_filter = get_filter(fs, config.low_cut_freq, config.high_cut_freq)
    signal = filter_signal(signal, sos_filter)
    signal = resample_signal(signal, fs, config.resample_freq)
    signal = ensure_signal_size(signal, input_length, config.random_edge_cut)

    return signal


def trim_zeros(signal: np.ndarray) -> np.ndarray:
    """Trim rows from the start and end where all channels (columns) are zero."""
    nonzero_rows = np.nonzero(np.any(signal != 0, axis=1))[0]
    return signal[nonzero_rows[0]:nonzero_rows[-1] + 1, :]


def get_filter(fs: int, low_cut_freq: float, high_cut_freq: float):
    global filters

    if not filters:
        filters = create_filters(low_cut_freq, high_cut_freq)

    try:
        return filters[fs]
    except KeyError:
        raise Exception(f'No filter defined for {fs=}. Add it to `POSSIBLE_FS` and try again')


def create_filters(low_cut_freq: float, high_cut_freq: float, filter_order: int = FILTER_ORDER) -> dict:
    filters = {}
    for fs in POSSIBLE_FS:
        nyquist_freq = 0.5 * fs
        low = low_cut_freq / nyquist_freq
        high = high_cut_freq / nyquist_freq
        filters[fs] = butter(filter_order, [low, high], btype="band", output='sos', analog=False)
    return filters


def ensure_signal_size(signal: np.ndarray, input_length: int, random_edge_cut: bool) -> np.ndarray:
    """
    Ensure the signal is of shape (input_length, 12) using center padding or truncation.
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected signal with 2 dimensions (time, channels), got shape {signal.shape}")

    signal = _adjust_length(signal, input_length, random_edge_cut)
    signal = _adjust_channels(signal, target_channels=12)
    return signal


def _adjust_length(signal: np.ndarray, target_length: int, random_edge_cut: bool) -> np.ndarray:
    current_length = signal.shape[0]
    if current_length > target_length:
        start = (current_length - target_length) // 2
        # 50% to be centered, 25% to be from start or end
        if random_edge_cut:
            if random.random() < 0.5:
                return signal[start:start + target_length]
            if random.random() < 0.5:
                return signal[:target_length]
            return signal[-target_length:]
        return signal[start:start + target_length]
    elif current_length < target_length:
        pad_total = target_length - current_length
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(signal, ((pad_before, pad_after), (0, 0)), mode='constant')
    return signal


def _adjust_channels(signal: np.ndarray, target_channels: int) -> np.ndarray:
    current_channels = signal.shape[1]
    if current_channels > target_channels:
        return signal[:, :target_channels]
    elif current_channels < target_channels:
        padded = np.zeros((signal.shape[0], target_channels), dtype=signal.dtype)
        padded[:, :current_channels] = signal
        return padded
    return signal


def resample_signal(original_signal: np.ndarray, original_fs: int, new_fs: int):
    num_samples = int(len(original_signal) * (new_fs / original_fs))
    return resample(original_signal, num_samples)


def filter_signal(signal: np.ndarray, sos_filter) -> np.ndarray:
    # not needed because HIGH_CUT is lower than 50, 60
    # pre = Preprocessing(signal, fs)
    # # 50 Hz for european powerline, 60 Hz for USA
    # pre.notch(50)
    # pre.notch(60)
    # signal = pre.signal

    return apply_band_pass_filter(signal, sos_filter)


def reorder_leads(signal: np.ndarray, signal_names: list[str]) -> np.ndarray:
    try:
        lead_order = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        index_map = [signal_names.index(lead) for lead in lead_order]
        return signal[:, index_map]
    except Exception:
        logger.exception('Could not reorder leads. Leaving as it is. Error:')
        return signal


def apply_band_pass_filter(signal: np.ndarray, sos_filter) -> np.ndarray:
    return np.apply_along_axis(lambda x: sosfiltfilt(sos_filter, x), axis=0, arr=signal)


def remove_baseline_wander(signal: np.ndarray, wavelet: str = 'db6') -> np.ndarray:
    """Use Wavelet transform to remove baseline wander"""
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    level = min(6, max_level)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])
    return pywt.waverec(coeffs, wavelet)[:len(signal)]
