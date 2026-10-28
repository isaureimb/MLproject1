import numpy as np
import matplotlib.pyplot as plt


def explore_dimensions(y_train, x_train, x_test=None):
    """Compute and display dataset dimensions, shapes, and class distribution.

    Args:
        y_train: shape=(N_train, 1) or (N_train,)
        x_train: shape=(N_train, D)
        x_test: shape=(N_test, D), optional

    Returns:
        N_train: number of training samples (int), number of rows
        D:number of features (int), number of columns
    """
    assert (
        y_train.shape[0] == x_train.shape[0]
    ), "y_train and x_train must have the same number of rows"
    if x_test is not None:
        assert (
            x_train.shape[1] == x_test.shape[1]
        ), "x_train and x_test must have the same number of features"

    N_train, D = x_train.shape
    classes, counts = np.unique(y_train, return_counts=True)

    print(f"Training samples  (N_train): {N_train}")
    print(f"Features          (D)      : {D}")
    print(f"Classes: {classes.tolist()} | Counts: {counts.tolist()}")

    if x_test is not None:
        N_test = x_test.shape[0]
        print(f"Test samples      (N_test) : {N_test}")

    return N_train, D


import numpy as np


def clean_special_codes_auto(
    x_train,
    x_test,
    candidate_codes=(7, 9, 77, 88, 99, 777, 999, 7777, 9999),
    gap_ratio=1.5,
):
    """Detect and replace BRFSS sentinel/special response codes with NaN.

    Decision rules per feature (computed strictly on x_train):
    1. candidate >= 77: considered an invalid sentinel if it exceeds the maximum
       of the other non-sentinel values by at least `gap_ratio` (e.g. 1.5x).
    2. candidate in (7, 9): considered a refusal code if the remaining values
       are strictly categorical on a narrow scale (max <= 5).

    Args:
        x_train: np.ndarray, shape=(N_train, D)
        x_test:  np.ndarray, shape=(N_test, D)
        candidate_codes: iterable of suspected BRFSS survey codes
        gap_ratio: float, threshold factor for the gap analysis

    Returns:
        x_train_clean: np.ndarray, shape=(N_train, D)
        x_test_clean:  np.ndarray, shape=(N_test, D)
    """

    # assert (x_train.ndim == 2 and x_test.ndim == 2), "x_train and x_test must be 2D arrays"
    # assert (x_train.shape[1] == x_test.shape[1]), "Feature count mismatch between train and test"

    x_train_clean = x_train.copy().astype(float)
    x_test_clean = x_test.copy().astype(float)

    n_features = x_train.shape[1]

    for j in range(n_features):
        col_train = x_train[:, j]
        # take only the valid values of the column
        valid_train = col_train[~np.isnan(col_train)]

        # if there is only NaN values in the column, skip it
        if len(valid_train) == 0:
            continue

        unique_vals = set(np.unique(valid_train))

        for code in candidate_codes:
            if code not in unique_vals:
                continue

            # values in the column that are not the current candidate code
            remaining = valid_train[valid_train != code]
            if len(remaining) == 0:
                continue

            max_rem = np.max(remaining)

            # rules for detecting sentinel codes
            is_large_gap = (code >= 77) and (code > max_rem * gap_ratio)
            is_categorical_refusal = (code in (7, 9)) and (max_rem <= 5)

            if is_large_gap or is_categorical_refusal:
                x_train_clean[x_train_clean[:, j] == code, j] = np.nan
                x_test_clean[x_test_clean[:, j] == code, j] = np.nan

    return x_train_clean, x_test_clean


def drop_useless_columns(x_train, x_test, missing_threshold=0.5, var_threshold=1e-8):
    """Drop columns with excessive NaN ratios or near-zero variance based on train set.

    Args:
        x_train: np.ndarray, shape=(N_train, D)
        x_test:  np.ndarray, shape=(N_test, D)
        missing_threshold: float, maximum allowed fraction of NaNs per column
        var_threshold: float, minimal variance required to keep a feature

    Returns:
        x_train_filtered: np.ndarray, shape=(N_train, D_filtered)
        x_test_filtered:  np.ndarray, shape=(N_test, D_filtered)
        keep_columns:     np.ndarray of bool, shape=(D,)
    """
    assert x_train.ndim == 2 and x_test.ndim == 2, "Inputs must be 2D arrays"
    assert (
        x_train.shape[1] == x_test.shape[1]
    ), "Feature count mismatch between train and test"

    N_train = x_train.shape[0]

    # Mask features exceeding the NaN ratio
    nan_ratios = (
        np.isnan(x_train).sum(axis=0) / N_train
    )  # np.isnan() is a boolean matrix
    valid_missing = nan_ratios <= missing_threshold

    # Mask constant or near-constant features
    variances = np.nanvar(x_train, axis=0)
    valid_variance = ~np.isnan(variances) & (
        variances > var_threshold
    )  # verify that columns is suficiently variable

    keep_columns = valid_missing & valid_variance

    return x_train[:, keep_columns], x_test[:, keep_columns], keep_columns


def impute_by_median(x_train, x_test=None):
    """Impute remaining NaNs using the median computed on x_train only.

    Args:
        x_train: np.ndarray, shape=(N_train, D)
        x_test:  np.ndarray, shape=(N_test, D), optional

    Returns:
        x_train_imp: np.ndarray, shape=(N_train, D)
        x_test_imp:  np.ndarray, shape=(N_test, D), if x_test is provided
    """
    # calculate the median of each column in x_train, ignoring NaNs
    medians = np.nanmedian(x_train, axis=0)

    # replace NaNs in x_train and x_test with the corresponding median values
    x_train_imp = np.where(np.isnan(x_train), medians, x_train)
    x_test_imp = np.where(np.isnan(x_test), medians, x_test)

    return x_train_imp, x_test_imp


def standardize(x_train, x_test):
    """Standardize features to have zero mean and unit variance.

    Mean and standard deviation are computed strictly on x_train
    and then applied to both x_train and x_test.

    Args:
        x_train: np.ndarray, shape=(N_train, D)
        x_test:  np.ndarray, shape=(N_test, D)

    Returns:
        x_train_std: np.ndarray, shape=(N_train, D)
        x_test_std:  np.ndarray, shape=(N_test, D)
    """
    # 1. calculate mean and std on x_train columns
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    # 2. Protection against division by zero (if std == 0)
    std = np.where(std < 1e-8, 1.0, std)

    # 3. Centering and scaling vectors
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    return x_train_std, x_test_std


def add_bias_term(x_train, x_test):
    """Add an intercept column of ones (bias) as the first column.

    Args:
        x_train: np.ndarray, shape=(N_train, D)
        x_test:  np.ndarray, shape=(N_test, D)

    Returns:
        tx_train: np.ndarray, shape=(N_train, D + 1)
        tx_test:  np.ndarray, shape=(N_test, D + 1)
    """
    ones_tr = np.ones((x_train.shape[0], 1))
    ones_te = np.ones((x_test.shape[0], 1))

    tx_train = np.hstack((ones_tr, x_train))
    tx_test = np.hstack((ones_te, x_test))

    return tx_train, tx_test


def most_correlated_features(x, y, n_top=30):
    """Rank and plot the features by the strength of their linear correlation with the target.

    Missing values (NaN) are ignored feature by feature.

    Args:
        x: numpy array of shape (N, D), features (may contain NaN)
        y: numpy array of shape (N,), labels in {0, 1}
        n_top: number of features to return
        feature_names: optional list of the D column names

    """
    corrs = np.zeros(x.shape[1])

    for j in range(x.shape[1]):
        col = x[:, j]
        mask = ~np.isnan(col)

        # Correlation is undefined if too few values or if one variable is constant
        if mask.sum() < 2 or np.std(col[mask]) == 0 or np.std(y[mask]) == 0:
            continue

        corrs[j] = np.corrcoef(col[mask], y[mask])[0, 1]

    top_idx = np.argsort(np.abs(corrs))[::-1][:n_top]
    top_corr = np.abs(corrs[top_idx])

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(top_idx)), top_corr)
    plt.xticks(range(len(top_idx)), top_idx, rotation=90)
    plt.xlabel("feature index")
    plt.ylabel("Correlation")
    plt.title("highest correlations")
    plt.show()


def preprocess(x_train, x_test, missing_threshold=0.5):
    """Apply the full preprocessing pipeline (statistics computed on train only).

    Returns:
        tx_train: shape=(N_train, D'+1), standardized, with bias column
        tx_test:  shape=(N_test, D'+1)
    """
    x_train, x_test = clean_special_codes_auto(x_train, x_test)
    x_train, x_test, _ = drop_useless_columns(x_train, x_test, missing_threshold)
    x_train, x_test = impute_by_median(x_train, x_test)
    x_train, x_test = standardize(x_train, x_test)
    return add_bias_term(x_train, x_test)
