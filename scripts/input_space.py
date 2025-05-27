from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from data_tools.data_set import ECGDataset
from data_tools.preprocess import preprocess
from helper_code import load_signals
from settings import Config, PreprocessConfig

CODE_15_PATH = '/MLdata/shai/physionet2025/code15'
PTBXL_PATH = '/MLdata/shai/physionet2025/ptbxl'
SAMITROP_PATH = '/MLdata/shai/physionet2025/samitrop'

CODE_15_SYMBOL = 'code15'
PTBXL_SYMBOL = 'ptbxl'
SAMITROP_SYMBOL = 'samitrop'

config = Config()


def get_inputs() -> tuple[np.ndarray, list]:
    """
    Returns shapes (N1+N2+N3, 12, 934) and (N1+N2+N3,)
    """
    inputs_list = []
    dataset_list = []

    print('Preprocessing SamiTrop')
    samitrop_inputs = get_dataset_inputs(SAMITROP_PATH)
    inputs_list.append(samitrop_inputs)
    dataset_list.extend([SAMITROP_SYMBOL] * samitrop_inputs.shape[0])

    print('Preprocessing PTB-XL')
    ptbxl_inputs = get_dataset_inputs(PTBXL_PATH)
    inputs_list.append(ptbxl_inputs)
    dataset_list.extend([PTBXL_SYMBOL] * samitrop_inputs.shape[0])

#    print('Preprocessing CODE-15%')
#    code_15_inputs = get_dataset_inputs(CODE_15_PATH)
#    inputs_list.append(code_15_inputs)
#    dataset_list.extend([CODE_15_SYMBOL] * samitrop_inputs.shape[0])

    return np.concatenate(inputs_list, axis=0), dataset_list


def get_dataset_inputs(folder_path: str) -> np.ndarray:
    """
    Returns shape (N, 12, 934)
    """
    dataset = ECGDataset(folder_path, config.data.input_length, config.pre_process)
    input_length = config.data.input_length
    pre_process = config.pre_process

    with ProcessPoolExecutor() as executor:
        process_func = partial(process_record, input_length=input_length, pre_process=pre_process)
        signals = list(executor.map(process_func, dataset.record_files))

    return np.stack(signals, axis=0)


def process_record(record_file_name: str, input_length: int, pre_process: PreprocessConfig) -> np.ndarray:
    signal, fields = load_signals(record_file_name)
    preprocessed = preprocess(signal, fields['sig_name'], fields['fs'], input_length, pre_process)
    return preprocessed.T


def reduce(x: np.ndarray, method: str = 'umap') -> np.ndarray:
    """
    (N, C) -> (N, 3)
    C is 934*12 (input space) or 1088 (feature space)
    """
    if method == 'umap':
        reducer = umap.UMAP(n_components=3, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=3, perplexity=30, random_state=42)
    else:
        raise Exception(f'{method=} is not supported')
    return reducer.fit_transform(x)


def plot(embeddings: np.ndarray, dataset_labels: list, title: str) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                    c=dataset_labels, alpha=0.8)
    ax.set_title(title)
    plt.grid(True)
    legend_elements = sc.legend_elements()[0]
    label_names = np.unique(dataset_labels)
    plt.legend(legend_elements, label_names, title="Dataset")
    # plt.show()
    plt.savefig(f'{title}.png', dpi=300)


def main() -> None:
    print('Getting inputs')
    x, dataset_labels = get_inputs()
    x_flat = x.reshape(x.shape[0], -1)
    print('Reducing')
    embeddings = reduce(x_flat)
    plot(embeddings, dataset_labels, 'input_space_3d_datasets')


if __name__ == '__main__':
    main()
