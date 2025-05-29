from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data_tools.data_set import ECGDataset
from data_tools.preprocess import preprocess
from helper_code import load_signals, load_label
from settings import Config, PreprocessConfig

CODE_15_PATH = '/MLdata/shai/physionet2025/code15'
PTBXL_PATH = '/MLdata/shai/physionet2025/ptbxl'
SAMITROP_PATH = '/MLdata/shai/physionet2025/samitrop'

CODE_15_SYMBOL = 'code15'
PTBXL_SYMBOL = 'ptbxl'
SAMITROP_SYMBOL = 'samitrop'
DATASET_SAMPLE_SIZE = 800

config = Config()
np.random.seed(4)


def get_inputs() -> tuple[np.ndarray, list, list]:
    """
    Returns shapes (N1+N2+N3, 12, 934) and (N1+N2+N3,)
    """
    inputs_list = []
    dataset_list = []
    chagas_labels = []

    print('Preprocessing SamiTrop')
    inputs, labels = get_dataset_inputs(SAMITROP_PATH)
    inputs_list.append(inputs)
    chagas_labels.extend(labels)
    dataset_list.extend([SAMITROP_SYMBOL] * inputs.shape[0])

    print('Preprocessing PTB-XL')
    inputs, labels = get_dataset_inputs(PTBXL_PATH)
    inputs_list.append(inputs)
    chagas_labels.extend(labels)
    dataset_list.extend([PTBXL_SYMBOL] * inputs.shape[0])

    print('Preprocessing CODE-15%')
    inputs, labels = get_dataset_inputs(CODE_15_PATH)
    inputs_list.append(inputs)
    chagas_labels.extend(labels)
    dataset_list.extend([CODE_15_SYMBOL] * inputs.shape[0])

    return np.concatenate(inputs_list, axis=0), dataset_list, chagas_labels


def get_dataset_inputs(folder_path: str) -> tuple[np.ndarray, list[int]]:
    """
    Returns shape (N, 12, 934) and a list of size N
    """
    dataset = ECGDataset(folder_path, config.data.input_length, config.pre_process)
    input_length = config.data.input_length
    pre_process = config.pre_process
    sampled_record_files = np.random.choice(dataset.record_files, size=DATASET_SAMPLE_SIZE, replace=False)

    with ProcessPoolExecutor() as executor:
        process_func = partial(process_record, input_length=input_length, pre_process=pre_process)
        results = list(executor.map(process_func, sampled_record_files))
        signals, labels = zip(*results)

    return np.stack(list(signals), axis=0), list(labels)


def process_record(record_file_name: str, input_length: int, pre_process: PreprocessConfig) -> tuple[np.ndarray, int]:
    signal, fields = load_signals(record_file_name)
    preprocessed = preprocess(signal, fields['sig_name'], fields['fs'], input_length, pre_process)
    return preprocessed.T, load_label(record_file_name)


def reduce(x: np.ndarray, method: str = 'umap') -> np.ndarray:
    print('Normalizing')
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    print('Reducing using PCA (intermediate)')
    pca = PCA(n_components=50, random_state=0)
    x_pca = pca.fit_transform(x_std)

    print('Performing final reduction')
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, n_jobs=64)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=0)
    else:
        raise Exception(f'{method=} is not supported')
    return reducer.fit_transform(x_pca)


def classify(embeddings: np.ndarray, labels: list) -> None:
    print('Training a classifier')
    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = RandomForestClassifier()
    scores = cross_val_score(clf, embeddings, y, cv=5)

    print("Classification accuracy:", scores.mean())
