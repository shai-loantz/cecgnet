from torch import Tensor, sigmoid

from helper_code import compute_auc, compute_challenge_score, compute_accuracy, compute_f_measure


def calculate_aggregate_metrics(y_pred: Tensor, y: Tensor, threshold: float) -> dict[str, Tensor]:
    labels = y.cpu().float().numpy()
    prob_outputs = sigmoid(y_pred).cpu().float().numpy()
    binary_outputs = (prob_outputs > threshold).astype(int)
    challenge_score = compute_challenge_score(labels, prob_outputs)
    auroc, auprc = compute_auc(labels, prob_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)
    return {'challenge_score': challenge_score, 'auroc': auroc, 'auprc': auprc, 'accuracy': accuracy,
            'f_measure': f_measure}
