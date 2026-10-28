import numpy as np


def compute_metrics(y_true, y_pred):
    """Compute classification metrics: Accuracy, Precision, Recall, and F1-score.

    Assumes binary labels in {-1, 1} or {0, 1}.

    Args:
        y_true: numpy array of shape (N,), true labels.
        y_pred: numpy array of shape (N,), predicted labels, same encoding as y_true.

    Returns:
        dict with keys "accuracy", "precision", "recall", "f1" (floats in [0, 1]).
        Precision, recall and F1 are set to 0.0 when their denominator is 0
        (e.g. no sample predicted positive).
    """

    y_t = (y_true == 1).astype(int)
    y_p = (y_pred == 1).astype(int)

    tp = np.sum((y_t == 1) & (y_p == 1))
    fp = np.sum((y_t == 0) & (y_p == 1))
    fn = np.sum((y_t == 1) & (y_p == 0))
    tn = np.sum((y_t == 0) & (y_p == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
