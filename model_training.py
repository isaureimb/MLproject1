"""
Conventions:

* y is converted to {0, 1} for every model (the course functions expect that);
  we only go back to {-1, 1} when writing the submission.
* A "model" is a pair of functions:
      fit(y01, tx)   -> w
      score(tx, w)   -> real-valued score (linear output or probability)
  A prediction is then ``score >= threshold``.
* Cross-validation is stratified (same % of sick people in every fold) and
  returns out-of-fold (OOF) scores, so the decision threshold can be tuned on them.
"""

import numpy as np
import matplotlib.pyplot as plt

from implementations import (
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
    weighted_logistic_gd,
    stable_sigmoid,
)
from metrics import compute_metrics
from cross_validation import (
    cross_validation,
    stratified_subsample,
)

SEED = 1
LOG_FILE = "results_log.txt"  # keep the scores of the models comparison in a text file


# Label helpers


def to_01(y):
    """Convert labels from {-1, 1} to {0, 1}."""
    return (y == 1).astype(float)


def to_pm1(y01):
    """Convert labels from {0, 1} back to {-1, 1}."""
    return np.where(y01 == 1, 1, -1)


# Threshold tuning (maximise F1)


def best_threshold_f1(y01, scores):
    """Find the decision threshold that maximises the F1-score.

    With imbalanced classes, the default threshold of 0.5 usually predicts
    too few positives. This function searches for the threshold that gives
    the best F1-score, where a sample is predicted positive if
    score >= threshold.

     Vectorised: sort by decreasing score; if we predict the top-k as positive,
    F1 = 2*TP / (k + P) where P is the total number of positives.

    Args:
        y01: numpy array of shape (N,), true labels in {0, 1}.
        scores: numpy array of shape (N,), predicted scores

    Returns:
        threshold: float, the best decision threshold.
        best_f1: float, the F1-score obtained with this threshold.
    """

    order = np.argsort(-scores)
    s, ys = scores[order], y01[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    i = int(np.argmax(f1))
    thr = (s[i] + s[i + 1]) / 2 if i < len(s) - 1 else s[i]
    return thr, f1[i]


# score
def linear_score(tx, w):
    """Score of a linear model (least squares, ridge): the linear output tx @ w.

    Args:
        tx: numpy array of shape (N, D), features.
        w: numpy array of shape (D,), weights.

    Returns:
        numpy array of shape (N,), one real-valued score per sample.
    """
    return tx @ w


def prob_score(tx, w):
    """Score of a logistic model: the predicted probability of class 1.

    Args:
        tx: numpy array of shape (N, D), features.
        w: numpy array of shape (D,), weights.

    Returns:
        numpy array of shape (N,), probabilities in (0, 1).
    """
    return stable_sigmoid(tx @ w)


# Models factories

"""
The functions of implementations.py have different signatures and return
(w, loss). To evaluate any model the same way in cross-validation, each
factory fixes the model's hyper-parameters and returns a pair (fit, score):
   fit(y, tx) -> w            weights only, shape (D,); labels in {0, 1}
   score(tx, w) -> scores     shape (N,): tx @ w (linear_score) or
                              sigmoid(tx @ w) (prob_score)
A sample is predicted positive if score >= threshold.
"""


def majority_model():
    """Always predicts the majority class (w = 0 -> constant score)."""
    return (lambda y, tx: np.zeros(tx.shape[1])), linear_score


def least_squares_model():
    def fit(y, tx):
        try:
            return least_squares(y, tx)[0]
        except np.linalg.LinAlgError:
            return ridge_regression(y, tx, 1e-8)[0]

    return fit, linear_score


def ridge_model(lambda_):
    return (lambda y, tx: ridge_regression(y, tx, lambda_)[0]), linear_score


def logistic_model(max_iters, gamma):
    def fit(y, tx):
        return logistic_regression(y, tx, np.zeros(tx.shape[1]), max_iters, gamma)[0]

    return fit, prob_score


def reg_logistic_model(lambda_, max_iters, gamma):
    def fit(y, tx):
        w0 = np.zeros(tx.shape[1])
        return reg_logistic_regression(y, tx, lambda_, w0, max_iters, gamma)[0]

    return fit, prob_score


def weighted_logistic_model(lambda_, max_iters, gamma, pos_weight):
    def fit(y, tx):
        return weighted_logistic_gd(y, tx, lambda_, max_iters, gamma, pos_weight)

    return fit, prob_score


# Models evaluation


def log_result(name, res, log_file=LOG_FILE):
    """Print the results of a model and append them to the log file.

    Args:
        name: str, name of the model and its hyper-parameters.
        res: dict of metrics, as returned by evaluate_model, with keys
            "acc@0.5", "f1@0.5" (at threshold 0.5), "thr" (best threshold)
            and "f1", "precision", "recall", "accuracy" (at the best threshold).
        log_file: str, path of the log file.
    """

    line = (
        f"{name} | thr=0.5: acc={res['acc@0.5']:.4f} f1={res['f1@0.5']:.4f} "
        f"| best thr={res['thr']:.4f}: f1={res['f1']:.4f} "
        f"prec={res['precision']:.4f} rec={res['recall']:.4f} acc={res['accuracy']:.4f}"
    )
    print(line)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def evaluate_model(name, fit, score, y01, tx, k_fold=5, seed=SEED, log=True):
    """Evaluate a model with stratified k-fold cross-validation.

    Note: the best threshold is chosen on the pooled OOF scores, so the F1 at
    that threshold is very slightly optimistic (1 parameter tuned on 328k points).

    Args:
        name: str, name of the model, used in the log.
        fit: function fit(y, tx) -> w, trains the model.
        score: function score(tx, w) -> scores (linear_score or prob_score).
        y01: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), preprocessed features.
        k_fold: int, number of folds.
        seed: int, random seed used to build the folds.
        log: bool, if True, print the results and append them to the log file.

    Returns:
        res: dict of metrics, with keys
            "acc@0.5", "f1@0.5": accuracy and F1 at threshold 0.5,
            "thr": best threshold,
            "accuracy", "precision", "recall", "f1": metrics at the best threshold.
        oof: numpy array of shape (N,), out-of-fold scores.
    """

    oof = cross_validation(y01, tx, k_fold, seed, fit, score)
    m05 = compute_metrics(y01, (oof >= 0.5).astype(int))
    thr, _ = best_threshold_f1(y01, oof)
    mbest = compute_metrics(y01, (oof >= thr).astype(int))
    res = {
        "acc@0.5": m05["accuracy"],
        "f1@0.5": m05["f1"],
        "thr": thr,
        **mbest,
    }
    if log:
        log_result(name, res)
    return res, oof


# baselines


def run_baselines(y01, tx, k_fold=5, lambdas=np.logspace(-6, 0, 7)):
    """Evaluate the baseline models with cross-validation.

    Models evaluated:
        - majority class (always predicts 0): shows that accuracy is
          misleading on imbalanced data,
        - least squares,
        - ridge regression, for each value in `lambdas`,
        - logistic regression (gamma=0.1, 300 iterations),
        - regularised logistic regression (lambda=1e-4, gamma=0.1, 300 iterations).

    Args:
        y01: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), preprocessed features.
        k_fold: int, number of folds.
        lambdas: iterable of floats, regularisation strengths tested for ridge.

    Returns:
        results: dict mapping each model name (e.g. "ridge_1e-04") to its
            dict of metrics, as returned by evaluate_model.
    """

    results = {}
    results["majority"], _ = evaluate_model(
        "majority", *majority_model(), y01, tx, k_fold
    )
    results["least_squares"], _ = evaluate_model(
        "least_squares", *least_squares_model(), y01, tx, k_fold
    )
    for lam in lambdas:
        results[f"ridge_{lam:.0e}"], _ = evaluate_model(
            f"ridge lambda={lam:.0e}", *ridge_model(lam), y01, tx, k_fold
        )
    results["logistic"], _ = evaluate_model(
        "logistic gamma=0.1 iters=300", *logistic_model(300, 0.1), y01, tx, k_fold
    )
    results["reg_logistic"], _ = evaluate_model(
        "reg_logistic lambda=1e-4 gamma=0.1 iters=300",
        *reg_logistic_model(1e-4, 300, 0.1),
        y01,
        tx,
        k_fold,
    )
    return results


def plot_baselines_summary(results, save_path="baselines_comparison.png"):
    """make a bar plot comparing the baseline models on accuracy and F1-score."""
    names = list(results.keys())
    acc_05 = [results[m]["acc@0.5"] for m in names]
    f1_05 = [results[m]["f1@0.5"] for m in names]
    f1_best = [results[m]["f1"] for m in names]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, acc_05, width, label="Accuracy (thr=0.5)", color="#4C72B0")
    plt.bar(x, f1_05, width, label="F1-score (thr=0.5)", color="#C44E52")
    plt.bar(x + width, f1_best, width, label="F1-score (best thr)", color="#55A868")

    plt.ylabel("Score")
    plt.title("Comparison of Baseline Models (5-fold CV)")
    plt.xticks(x, names, rotation=35, ha="right", fontsize=9)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Graph saved as : {save_path}")
    plt.show()


# grid search


def grid_search_weighted_logistic(
    y01,
    tx,
    lambdas=np.logspace(-6, -1, 6),
    gammas=(0.1, 0.3),
    pos_weights=(1.0, 3.0, 8.0),
    max_iters=300,
    k_fold=5,
    n_max=60000,
):
    """Grid search for the (class-weighted) regularised logistic regression.

    Each config is scored by the best-F1 over OOF scores. `n_max` runs the search
    on a stratified subsample to save time (set to None to use everything).

    Args:
        y01: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), preprocessed features.
        lambdas: iterable of floats, regularisation strengths to test.
        gammas: iterable of floats, step sizes to test.
        pos_weights: iterable of floats, weights of the positive class to test.
        max_iters: int, number of gradient descent steps (fixed).
        k_fold: int, number of folds.
        n_max: int or None, size of the subsample (None = use all the data).

    Returns:
        best_cfg: dict with keys "lambda_", "gamma", "pos_weight", "max_iters",
            the configuration with the highest F1.
        all_res: list of (cfg, f1) tuples, one per tested configuration.
    """
    sub = stratified_subsample(y01, n_max)
    y_s, tx_s = y01[sub], tx[sub]
    all_res, best = [], None
    for lam in lambdas:
        for gamma in gammas:
            for pw in pos_weights:
                fit, score = weighted_logistic_model(lam, max_iters, gamma, pw)
                res, _ = evaluate_model(
                    f"wlogistic lambda={lam:.0e} gamma={gamma} pos_w={pw}",
                    fit,
                    score,
                    y_s,
                    tx_s,
                    k_fold,
                )
                cfg = dict(lambda_=lam, gamma=gamma, pos_weight=pw, max_iters=max_iters)
                all_res.append((cfg, res["f1"]))
                if best is None or res["f1"] > best[1]:
                    best = (cfg, res["f1"], res["thr"])
    print(f"\nBest config: {best[0]} | F1={best[1]:.4f} | thr={best[2]:.4f}")
    return best[0], all_res


# Final fit + prediction


def fit_and_predict(cfg, y01, tx_train, tx_test, k_fold=5):
    """Train the final model and predict the labels of the test set.

    Steps:
        1. build the class-weighted logistic regression with the given
           hyper-parameters,
        2. choose the decision threshold on out-of-fold scores (k-fold CV on
           the full training set), so that it is tuned on unseen data,
        3. retrain the model on ALL the training data,
        4. predict the test set with this threshold.

    Args:
        cfg: dict of hyper-parameters, with keys "lambda_", "gamma",
            "pos_weight", "max_iters" (e.g. as returned by
            grid_search_weighted_logistic).
        y01: numpy array of shape (N_train,), training labels in {0, 1}.
        tx_train: numpy array of shape (N_train, D), preprocessed training features.
        tx_test: numpy array of shape (N_test, D), test features, preprocessed
            in exactly the same way as tx_train.
        k_fold: int, number of folds used to choose the threshold.

    Returns:
        numpy array of shape (N_test,), predicted labels in {-1, 1},
    """

    fit, score = weighted_logistic_model(
        cfg["lambda_"], cfg["max_iters"], cfg["gamma"], cfg["pos_weight"]
    )
    oof = cross_validation(y01, tx_train, k_fold, SEED, fit, score)
    thr, f1 = best_threshold_f1(y01, oof)
    print(f"CV F1 = {f1:.4f} at threshold {thr:.4f}")
    w = fit(y01, tx_train)
    return to_pm1((score(tx_test, w) >= thr).astype(int))
