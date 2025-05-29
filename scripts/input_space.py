from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib.colors as mcolors
import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
DATASET_SAMPLE_SIZE = 1000

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
    dataset_list.extend([PTBXL_SYMBOL] * ptbxl_inputs.shape[0])

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
    sampled_record_files = np.random.choice(dataset.record_files, size=DATASET_SAMPLE_SIZE, replace=False)

    with ProcessPoolExecutor() as executor:
        process_func = partial(process_record, input_length=input_length, pre_process=pre_process)
        signals = list(executor.map(process_func, sampled_record_files))

    return np.stack(signals, axis=0)


def process_record(record_file_name: str, input_length: int, pre_process: PreprocessConfig) -> np.ndarray:
    signal, fields = load_signals(record_file_name)
    preprocessed = preprocess(signal, fields['sig_name'], fields['fs'], input_length, pre_process)
    return preprocessed.T


def reduce(x: np.ndarray, method: str = 'umap') -> np.ndarray:
    print('Normalizing')
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    print('Reducing using PCA (intermediate)')
    pca = PCA(n_components=50, random_state=0)
    x_pca = pca.fit_transform(x_std)

    print('Performing final reduction')
    if method == 'umap':
        reducer = umap.UMAP(n_components=3, random_state=0)
    elif method == 'tsne':
        reducer = TSNE(n_components=3, perplexity=30, random_state=0)
    else:
        raise Exception(f'{method=} is not supported')
    return reducer.fit_transform(x_pca)


def plot(embeddings: np.ndarray, labels: list, title: str) -> None:
    unique_labels = np.unique(labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_index[label] for label in labels])

    colors = ['tab:blue', 'tab:green', 'tab:orange']  # Choose N colors for N classes
    cmap = mcolors.ListedColormap(colors[:len(unique_labels)])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
               c=int_labels, cmap=cmap, alpha=0.5, s=5)
    ax.set_title(title)
    plt.grid(True)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=label, markerfacecolor=colors[idx], markersize=8)
        for label, idx in label_to_index.items()
    ]
    plt.legend(handles=handles, title="Dataset")
    plt.savefig(f'{title}.png', dpi=300)
    # plt.show()


def main() -> None:
    print('Getting inputs')
    x, dataset_labels = get_inputs()
    x_flat = x.reshape(x.shape[0], -1)
    embeddings = reduce(x_flat, 'tsne')
    plot(embeddings, dataset_labels, 'input_space_3d_datasets')


if __name__ == '__main__':
    main()
