import numpy as np
import pywt
from scipy.signal import resample, butter, sosfiltfilt

from settings import PreprocessConfig
from utils.logger import logger

POSSIBLE_FS = (400, 500)  # Sampling frequencies of our known datasets
FILTER_ORDER = 10
filters: dict = {}


def preprocess(signal: np.ndarray, signal_names: list[str], fs: int, input_length: int, config: PreprocessConfig) -> np.ndarray:
    # signal = trim_leading_zeros(signal)  # TODO: what are we doing with padding?
    signal = reorder_leads(signal, signal_names)
    signal = np.apply_along_axis(remove_baseline_wander, 0, signal)
    sos_filter = get_filter(fs, config.low_cut_freq, config.high_cut_freq)
    signal = filter_signal(signal, sos_filter)
    signal = resample_signal(signal, fs, config.resample_freq)
    signal = ensure_signal_size(signal, input_length)

    return signal


def trim_leading_zeros(signal: np.ndarray) -> np.ndarray:
    """Find the first time (row) when not all leads (columns) are zero and start the signal there"""
    nonzero_rows = np.nonzero(np.any(signal != 0, axis=1))[0]
    return signal[nonzero_rows[0]:, :]


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


def ensure_signal_size(signal: np.ndarray, input_length: int) -> np.ndarray:
    """
    The signal should be of shape (input_length, 12).
    Pad or truncate (in time or channels) if shorter or longer.
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected signal with 2 dimensions (time, channels), got shape {signal.shape}")

    orig_length, orig_channels = signal.shape
    padded_signal = np.zeros((input_length, 12), dtype=signal.dtype)
    length_to_copy = min(orig_length, input_length)
    channels_to_copy = min(orig_channels, 12)
    padded_signal[:length_to_copy, :channels_to_copy] = signal[:length_to_copy, :channels_to_copy]

    return padded_signal


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
