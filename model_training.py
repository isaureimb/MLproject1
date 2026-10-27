"""Step 4-5: baselines, class-imbalance handling and hyper-parameter search.

Conventions
-----------
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

from implementations import (
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from metrics import compute_metrics
from cross_validation import cross_validation

SEED = 1
LOG_FILE = "results_log.txt"


# ----------------------------------------------------------------------
# Label helpers

def to_01(y):
    """{-1, 1} -> {0, 1}."""
    return (y == 1).astype(float)


def to_pm1(y01):
    """{0, 1} -> {-1, 1}."""
    return np.where(y01 == 1, 1, -1)


# ----------------------------------------------------------------------
# Cross-validation (stratified, NumPy only)


def stratified_subsample(y01, n_max, seed=SEED):
    """Indices of a stratified subsample of size ~n_max (to speed up grid searches)."""
    if n_max is None or n_max >= len(y01):
        return np.arange(len(y01))
    rng = np.random.RandomState(seed)
    frac = n_max / len(y01)
    idx = []
    for cls in (0, 1):
        c = np.where(y01 == cls)[0]
        idx.append(rng.choice(c, size=int(len(c) * frac), replace=False))
    return np.sort(np.concatenate(idx))

# ----------------------------------------------------------------------
# Threshold tuning (maximise F1)

def best_threshold_f1(y01, scores):
    """Find the threshold maximising F1 (prediction = score >= threshold).

    Vectorised: sort by decreasing score; if we predict the top-k as positive,
    F1 = 2*TP / (k + P) where P is the total number of positives.
    """
    order = np.argsort(-scores)
    s, ys = scores[order], y01[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    i = int(np.argmax(f1))
    thr = (s[i] + s[i + 1]) / 2 if i < len(s) - 1 else s[i]
    return thr, f1[i]


# ----------------------------------------------------------------------
# Models: (fit, score) factories

def stable_sigmoid(t):
    """Sigmoid without overflow warnings."""
    return 1.0 / (1.0 + np.exp(-np.clip(t, -30, 30)))


def linear_score(tx, w):
    return tx @ w


def prob_score(tx, w):
    return stable_sigmoid(tx @ w)


def majority_model():
    """Always predicts the majority class (w = 0 -> constant score)."""
    return (lambda y, tx: np.zeros(tx.shape[1])), linear_score


def least_squares_model():
    def fit(y, tx):
        try:
            return least_squares(y, tx)[0]
        except np.linalg.LinAlgError:  # singular tx.T @ tx (e.g. collinear one-hot)
            return ridge_regression(y, tx, 1e-8)[0]

    return fit, linear_score


def ridge_model(lambda_):
    return (lambda y, tx: ridge_regression(y, tx, lambda_)[0]), linear_score


def logistic_model(max_iters, gamma):
    # np.zeros created at each call: the course functions modify initial_w in place
    def fit(y, tx):
        return logistic_regression(y, tx, np.zeros(tx.shape[1]), max_iters, gamma)[0]

    return fit, prob_score


def reg_logistic_model(lambda_, max_iters, gamma):
    def fit(y, tx):
        w0 = np.zeros(tx.shape[1])
        return reg_logistic_regression(y, tx, lambda_, w0, max_iters, gamma)[0]

    return fit, prob_score


def weighted_logistic_gd(y, tx, lambda_, max_iters, gamma, pos_weight):
    """Logistic regression (GD) with a larger weight on the positive class.

    Loss = mean_i c_i * NLL_i + lambda_ * ||w||^2, with c_i = pos_weight if y_i = 1
    else 1 (normalised so that mean(c) = 1, which keeps gamma on the same scale).
    The bias (w[0]) is not regularised.
    """
    c = np.where(y == 1, pos_weight, 1.0)
    c = c / c.mean()
    w = np.zeros(tx.shape[1])
    reg_mask = np.ones_like(w)
    reg_mask[0] = 0.0
    for _ in range(max_iters):
        grad = tx.T @ (c * (stable_sigmoid(tx @ w) - y)) / len(y)
        grad += 2 * lambda_ * reg_mask * w
        w = w - gamma * grad
    return w


def weighted_logistic_model(lambda_, max_iters, gamma, pos_weight):
    def fit(y, tx):
        return weighted_logistic_gd(y, tx, lambda_, max_iters, gamma, pos_weight)

    return fit, prob_score


def with_undersampling(fit, ratio=1.0, seed=SEED):
    """Wrap a fit function: keep all positives and `ratio` x as many negatives.

    Applied INSIDE each training fold only (never on validation data).
    """

    def new_fit(y, tx):
        rng = np.random.RandomState(seed)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        neg = rng.choice(neg, size=min(len(neg), int(ratio * len(pos))), replace=False)
        idx = np.concatenate([pos, neg])
        return fit(y[idx], tx[idx])

    return new_fit


# ----------------------------------------------------------------------
# Evaluation + logging (feeds the ablation study)

def log_result(name, res, log_file=LOG_FILE):
    line = (
        f"{name} | thr=0.5: acc={res['acc@0.5']:.4f} f1={res['f1@0.5']:.4f} "
        f"| best thr={res['thr']:.4f}: f1={res['f1']:.4f} "
        f"prec={res['precision']:.4f} rec={res['recall']:.4f} acc={res['accuracy']:.4f}"
    )
    print(line)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def evaluate_model(name, fit, score, y01, tx, k_fold=5, seed=SEED, log=True):
    """CV a model; report metrics at the default threshold (0.5) and at the best-F1 one.

    Note: the best threshold is chosen on the pooled OOF scores, so the F1 at
    that threshold is very slightly optimistic (1 parameter tuned on 328k points).
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


# ----------------------------------------------------------------------
# Step 4: baselines

def run_baselines(y01, tx, k_fold=5, lambdas=np.logspace(-6, 0, 7)):
    """Majority, least squares, ridge, logistic, regularised logistic."""
    results = {}
    results["majority"], _ = evaluate_model("majority", *majority_model(), y01, tx, k_fold)
    results["least_squares"], _ = evaluate_model(r
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


# ----------------------------------------------------------------------
# Step 5: imbalance handling + grid search (hyper-parameters by CV)

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

    Returns:
        best config (dict), list of all (config, f1)
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


def compare_imbalance_strategies(y01, tx, cfg, k_fold=5):
    """Mini ablation on the full data: baseline vs weights vs undersampling."""
    lam, gamma, it = cfg["lambda_"], cfg["gamma"], cfg["max_iters"]
    evaluate_model(
        "[ablation] reg_logistic, thr=0.5 only meaningful",
        *reg_logistic_model(lam, it, gamma),
        y01,
        tx,
        k_fold,
    )
    evaluate_model(
        f"[ablation] weighted logistic pos_w={cfg['pos_weight']}",
        *weighted_logistic_model(lam, it, gamma, cfg["pos_weight"]),
        y01,
        tx,
        k_fold,
    )
    fit, score = reg_logistic_model(lam, it, gamma)
    evaluate_model(
        "[ablation] reg_logistic + undersampling 1:1",
        with_undersampling(fit, 1.0),
        score,
        y01,
        tx,
        k_fold,
    )


# ----------------------------------------------------------------------
# Final fit + prediction (used by run.py)

def fit_and_predict(cfg, y01, tx_train, tx_test, k_fold=5):
    """Tune the threshold on OOF scores, retrain on ALL train, predict test in {-1, 1}."""
    fit, score = weighted_logistic_model(
        cfg["lambda_"], cfg["max_iters"], cfg["gamma"], cfg["pos_weight"]
    )
    oof = cross_validation(y01, tx_train, k_fold, SEED, fit, score)
    thr, f1 = best_threshold_f1(y01, oof)
    print(f"CV F1 = {f1:.4f} at threshold {thr:.4f}")
    w = fit(y01, tx_train)
    return to_pm1((score(tx_test, w) >= thr).astype(int))


