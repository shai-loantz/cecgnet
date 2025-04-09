import json
import os
from glob import glob

import numpy as np
from torch import Tensor

from helper_code import compute_auc, compute_challenge_score, compute_accuracy, compute_f_measure
from settings import Config
from utils.logger import logger

OUTPUTS_DIR = '../outputs'
METRIC_NAMES = {'challenge_score', 'auroc', 'auprc', 'accuracy', 'f_measure'}


def write_outputs(rank: int, epoch: int, y_pred: Tensor, y: Tensor) -> None:
    """
    output file is a json of a list[tuple[list[float], list[float]]].
    The structure is - list of epochs,
                       for each epoch, we have a tuple where:
                       the first element is a list of y and the second is a list of y_pred
    """
    y_list = y.view(-1).cpu().tolist()
    y_pred_list = y_pred.view(-1).cpu().tolist()

    file_name = os.path.join(OUTPUTS_DIR, f'outputs_{rank}.json')
    epochs = _get_current_outputs(file_name)

    if len(epochs) == epoch + 1:
        epochs[epoch][0].extend(y_list)
        epochs[epoch][1].extend(y_pred_list)
    else:
        epochs.append((y_list, y_pred_list))

    with open(file_name, 'w') as fh:
        json.dump(epochs, fh)


def calculate_metrics_per_epoch() -> dict[str, list[float]]:
    logger.info('Aggregating outputs')
    epochs = _aggregate_outputs()

    metrics: dict[str, list[float]] = {}
    for metric_name in METRIC_NAMES:
        metrics[metric_name] = []

    config = Config()
    threshold = config.model.threshold

    for epoch in epochs:
        epoch_metrics = _calculate_metrics(np.array(epoch[0]), np.array(epoch[1]), threshold)
        for metric_name, value in epoch_metrics.items():
            metrics[metric_name].append(value)

    return metrics


def _calculate_metrics(labels: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    prob_outputs = _sigmoid(y_pred)
    binary_outputs = (prob_outputs > threshold).astype(int)
    challenge_score = compute_challenge_score(labels, prob_outputs)
    auroc, auprc = compute_auc(labels, prob_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)
    return {'challenge_score': challenge_score, 'auroc': auroc,
            'auprc': auprc, 'accuracy': accuracy, 'f_measure': f_measure}


def _aggregate_outputs() -> list[tuple[list[float], list[float]]]:
    first = True
    epochs: list[tuple[list[float], list[float]]] = []
    for file_name in glob(os.path.join(OUTPUTS_DIR, f'outputs_*.json')):
        rank_epochs = _get_current_outputs(file_name)
        if first:
            epochs = rank_epochs
        else:
            for epoch, outputs in enumerate(rank_epochs):
                epochs[epoch][0].extend(outputs[0])
                epochs[epoch][1].extend(outputs[1])
    return epochs


def _get_current_outputs(file_name: str) -> list[tuple[list[float], list[float]]]:
    epochs: list[tuple[list[float], list[float]]] = []
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if os.path.isfile(file_name):
        with open(file_name, 'r') as fh:
            epochs = json.load(fh)

    return epochs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0 + np.exp(-x))
