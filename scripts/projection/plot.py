import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def prepare_labels(labels):
    unique_labels = np.unique(labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_index[label] for label in labels])
    return int_labels, label_to_index


def create_legend(ax, label_to_index, color_list, legend_title):
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=label, markerfacecolor=color_list[idx], markersize=8)
        for label, idx in label_to_index.items()
    ]
    ax.legend(handles=handles, title=legend_title, loc='best')


def plot(embeddings: np.ndarray, dataset_labels: list, classification_labels: list, title: str) -> None:
    # Prepare dataset labels
    dataset_int_labels, dataset_label_map = prepare_labels(dataset_labels)
    dataset_colors = [plt.cm.tab10(i) for i in range(len(dataset_label_map))]
    dataset_cmap = mcolors.ListedColormap(dataset_colors)

    # Prepare classification labels
    class_int_labels, class_label_map = prepare_labels(classification_labels)
    class_colors = [plt.cm.tab20(i) for i in range(len(class_label_map))]
    class_cmap = mcolors.ListedColormap(class_colors)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Dataset-based 2D plot with colors
    sc1 = ax1.scatter(embeddings[:, 0], embeddings[:, 1], c=dataset_int_labels,
                      cmap=dataset_cmap, alpha=0.6, s=5)
    ax1.set_title("Colored by Dataset")
    ax1.grid(True)
    create_legend(ax1, dataset_label_map, dataset_colors, "Dataset")

    # Classification-based 2D plot with colors
    sc2 = ax2.scatter(embeddings[:, 0], embeddings[:, 1], c=class_int_labels,
                      cmap=class_cmap, alpha=0.6, s=5)
    ax2.set_title("Colored by Classification")
    ax2.grid(True)
    create_legend(ax2, class_label_map, class_colors, "Classification")

    # Save
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # prevent cropping
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
