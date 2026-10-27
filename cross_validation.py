import numpy as np


def build_k_indices(y, k_fold, seed=1):
    """Split the sample indices into k folds, keeping the class proportions.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1} or {0, 1}
        k_fold: number of folds
        seed: random seed, for reproducibility

    Returns:
        list of k_fold numpy arrays, each containing the indices of one fold
    """
    rng = np.random.RandomState(seed)
    positive = y == 1
    folds = [[] for _ in range(k_fold)]
    for mask in (positive, ~positive):
        idx = rng.permutation(np.where(mask)[0])
        # split each class separately so every fold gets the same class ratio
        for k, chunk in enumerate(np.array_split(idx, k_fold)):
            folds[k].append(chunk)
    return [np.concatenate(f) for f in folds]


def cross_validation(y, x, k_fold, seed, model_train, score_function):
    """Stratified k-fold cross-validation returning out-of-fold scores.

    Each sample is scored by a model that never saw it during training, so the
    decision threshold can then be tuned on these scores.

    Args:
        y:              shape=(N,), labels
        x:              shape=(N, D), feature matrix
        k_fold:         number of folds
        seed:           random seed
        model_train:    function (y_train, x_train) -> w
        score_function: function (x, w) -> real-valued scores, shape (N,)

    Returns:
        oof: shape=(N,), out-of-fold scores
    """
    k_indices = build_k_indices(y, k_fold, seed)
    oof = np.zeros(len(y))
    for k in range(k_fold):
        te_indice = k_indices[k]
        tr_indice = np.concatenate([k_indices[j] for j in range(k_fold) if j != k])
        w = model_train(y[tr_indice], x[tr_indice])
        oof[te_indice] = score_function(x[te_indice], w)
    return oof
