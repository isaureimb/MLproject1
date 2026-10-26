import numpy as np


def compute_metrics(y_true, y_pred):
    """Compute classification metrics: Accuracy, Precision, Recall, and F1-score.

    Assumes binary labels in {-1, 1} or {0, 1}.
    """
    # Si les labels sont {-1, 1}, on les ramène à {0, 1} pour faciliter les calculs booléens
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