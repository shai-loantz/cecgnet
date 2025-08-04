from scripts.projection.plot import plot
from scripts.projection.tools import get_inputs, reduce, classify


def main() -> None:
    print('Getting inputs')
    x, dataset_labels, chagas_labels = get_inputs()
    x_flat = x.reshape(x.shape[0], -1)
    embeddings = reduce(x_flat, 'tsne')
    plot(embeddings, dataset_labels, chagas_labels, 'input_space_2d_datasets')
    print('dataset classification:')
    classify(embeddings, dataset_labels)
    print('Chagas classification:')
    classify(embeddings, chagas_labels)


if __name__ == '__main__':
    main()
