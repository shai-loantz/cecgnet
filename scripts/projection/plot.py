"""Beware: A lot of ChatGPT code here"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def prepare_labels(labels):
    unique_labels = np.unique(labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_index[label] for label in labels])
    return int_labels, label_to_index


def create_color_legend(ax, label_to_index, colors, legend_title):
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=label, markerfacecolor=colors[idx], markersize=8)
        for label, idx in label_to_index.items()
    ]
    ax.legend(handles=handles, title=legend_title)


def create_marker_legend(ax, label_to_index, markers, legend_title):
    handles = [
        plt.Line2D([0], [0], marker=markers[idx], color='k',
                   label=label, linestyle='None', markersize=8)
        for label, idx in label_to_index.items()
    ]
    ax.legend(handles=handles, title=legend_title)


def plot(embeddings: np.ndarray, dataset_labels: list, classification_labels: list, title: str) -> None:
    # Prepare dataset labels
    dataset_int_labels, dataset_label_map = prepare_labels(dataset_labels)
    dataset_colors = [plt.cm.tab10(i) for i in range(len(dataset_label_map))]
    dataset_cmap = mcolors.ListedColormap(dataset_colors)

    # Prepare classification labels
    class_int_labels, class_label_map = prepare_labels(classification_labels)
    marker_styles = ['o', 's', '^', 'P', '*', 'X', 'D', 'v', '<', '>']
    if len(class_label_map) > len(marker_styles):
        raise ValueError("Too many classification labels for available markers.")
    class_markers = {i: marker_styles[i] for i in range(len(class_label_map))}

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Dataset-based 2D plot with colors
    sc1 = ax1.scatter(embeddings[:, 0], embeddings[:, 1], c=dataset_int_labels,
                      cmap=dataset_cmap, alpha=0.6, s=5)
    ax1.set_title("Colored by Dataset")
    ax1.grid(True)
    create_color_legend(ax1, dataset_label_map, dataset_colors, "Dataset")

    # Classification-based 2D plot with different markers
    for idx, label in enumerate(class_label_map):
        mask = class_int_labels == idx
        ax2.scatter(embeddings[mask, 0], embeddings[mask, 1],
                    marker=class_markers[idx], label=label, color='black', s=5, alpha=0.6)
    ax2.set_title("Shaped by Classification")
    ax2.grid(True)
    create_marker_legend(ax2, class_label_map, class_markers, "Classification")

    # Save
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.3)
    plt.savefig(f"{title}.png", dpi=300)
    plt.close()
