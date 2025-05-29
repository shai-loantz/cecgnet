import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def prepare_labels(labels):
    unique_labels = np.unique(labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_index[label] for label in labels])
    return int_labels, label_to_index


def create_legend(ax, label_to_index, colors, legend_title):
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=label, markerfacecolor=colors[idx], markersize=8)
        for label, idx in label_to_index.items()
    ]
    ax.legend(handles=handles, title=legend_title)


def plot(embeddings: np.ndarray, dataset_labels: list, chagas_labels: list, title: str) -> None:
    max_classes = max(len(np.unique(dataset_labels)), len(np.unique(chagas_labels)))
    cmap = plt.cm.get_cmap('tab10', max_classes)
    colors = [cmap(i) for i in range(max_classes)]
    color_map = mcolors.ListedColormap(colors)

    dataset_int_labels, dataset_label_map = prepare_labels(dataset_labels)
    class_int_labels, class_label_map = prepare_labels(chagas_labels)

    fig = plt.figure(figsize=(18, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                c=dataset_int_labels, cmap=color_map, alpha=0.5, s=5)
    ax1.set_title("Colored by Dataset")
    ax1.grid(True)
    create_legend(ax1, dataset_label_map, colors, "Dataset")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                c=class_int_labels, cmap=color_map, alpha=0.5, s=5)
    ax2.set_title("Colored by Chagas")
    ax2.grid(True)
    create_legend(ax2, class_label_map, colors, "Chagas Label")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
