import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from data_tools.data_set import ECGDataset
from data_tools.preprocess import preprocess
from helper_code import load_signals
from settings import Config

CODE_15_PATH = ''
PTBXL_PATH = ''
SAMITROP_PATH = ''

CODE_15_SYMBOL = 0
PTBXL_SYMBOL = 1
SAMITROP_SYMBOL = 2

config = Config()


def get_inputs() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns shapes (N1+N2+N3, 12, 934) and (N1+N2+N3,)
    """
    inputs_list = []
    dataset_list = []

    samitrop_inputs = get_dataset_inputs(SAMITROP_PATH)
    inputs_list.append(samitrop_inputs)
    dataset_list.append(np.ones_like(samitrop_inputs) * SAMITROP_SYMBOL)

    ptbxl_inputs = get_dataset_inputs(PTBXL_PATH)
    inputs_list.append(ptbxl_inputs)
    dataset_list.append(np.ones_like(ptbxl_inputs) * PTBXL_SYMBOL)

    code_15_inputs = get_dataset_inputs(CODE_15_PATH)
    inputs_list.append(code_15_inputs)
    dataset_list.append(np.ones_like(code_15_inputs) * CODE_15_SYMBOL)

    return np.concatenate(inputs_list, axis=0), np.concatenate(dataset_list, axis=0)


def get_dataset_inputs(folder_path: str) -> np.ndarray:
    """
    Returns shape (N, 12, 934)
    """
    dataset = ECGDataset(folder_path, config.input_length, config.preprocess_config)
    signals = []
    for record_file_name in dataset.record_files:
        signal, fields = load_signals(record_file_name)
        preprocessed = preprocess(signal, fields['sig_name'], fields['fs'], config.input_length, config.pre)
        signals.append(preprocessed.T)

    return np.stack(signals, axis=0)


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


def plot(embeddings: np.ndarray, dataset_labels: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                    c=dataset_labels, cmap='tab10', alpha=0.8)
    ax.set_title(title)
    plt.colorbar(sc)
    # plt.show()
    plt.savefig(f'{title}.png', dpi=300)


def main() -> None:
    x, dataset_labels = get_inputs()
    x_flat = x.reshape(x.shape[0], -1)
    print(f'{x.shape=}, {x_flat.shape=}, {dataset_labels.shape=}')
    embeddings = reduce(x)
    print(f'{embeddings.shape=}')
    plot(embeddings, dataset_labels, 'input_space_3d_datasets')


if __name__ == '__main__':
    main()
