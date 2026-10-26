"""
Entraînement, validation croisée et sélection d'hyperparamètres.

à valider avec preprocessing :
    tx_train : np.ndarray (N, D)   features nettoyées, standardisées, sans NaN,
                                   avec colonne de biais (1.0) déjà ajoutée
    y_train  : np.ndarray (N,)     labels en {-1, +1}
    tx_test  : np.ndarray (M, D)   mêmes transformations que le train
    ids_test : np.ndarray (M,)     identifiants du test (pour la soumission)

"""
import numpy as np

from implementations import ridge_regression, least_squares

# --------------------------------------------------------------------------
# Utilitaires labels / métriques
# --------------------------------------------------------------------------

def to_01(y):
    """{-1, +1} -> {0, 1} (nécessaire pour la régression logistique)."""
    return (y + 1) / 2


def to_pm1(y01):
    """{0, 1} -> {-1, +1} (format attendu par la soumission)."""
    return 2 * y01 - 1


def metrics(y_true01, y_pred01):
    """Accuracy, precision, recall, F1 (classe positive = malade)."""
    tp = np.sum((y_pred01 == 1) & (y_true01 == 1))
    fp = np.sum((y_pred01 == 1) & (y_true01 == 0))
    fn = np.sum((y_pred01 == 0) & (y_true01 == 1))
    acc = np.mean(y_pred01 == y_true01)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def best_threshold(scores, y01, n_thresholds=60):
    """Cherche le seuil sur les scores qui maximise le F1.

    Les données sont très déséquilibrées (~9% de positifs), donc le seuil par
    défaut (0.5 ou 0) est quasiment toujours sous-optimal pour le F1.
    On balaie les quantiles 50% -> 99.5% des scores.
    """
    candidates = np.quantile(scores, np.linspace(0.5, 0.995, n_thresholds))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = metrics(y01, (scores >= t).astype(int))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1

# --------------------------------------------------------------------------
# Modèles (wrappers autour de implementations.py + variante pondérée)
# --------------------------------------------------------------------------

def stable_sigmoid(t):
    """Sigmoïde sans overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(t, -30, 30)))


def weighted_reg_logistic(y01, tx, lambda_, initial_w, max_iters, gamma, pos_weight=1.0):
    """Régression logistique régularisée L2 avec poids sur la classe positive.

    Modification "maison" de reg_logistic_regression : chaque exemple positif
    compte pos_weight fois plus, ce qui compense le déséquilibre des classes.
    Retourne (w, loss) ; la loss est la NLL non pondérée, sans pénalité.
    """
    w = initial_w.copy()  # ne pas modifier initial_w en place
    sample_w = np.where(y01 == 1, pos_weight, 1.0)
    n = len(y01)
    for _ in range(max_iters):
        pred = stable_sigmoid(tx @ w)
        grad = tx.T @ (sample_w * (pred - y01)) / n + 2 * lambda_ * w
        w = w - gamma * grad
    pred = np.clip(stable_sigmoid(tx @ w), 1e-12, 1 - 1e-12)
    loss = -np.mean(y01 * np.log(pred) + (1 - y01) * np.log(1 - pred))
    return w, loss


def fit_ridge(y_pm1, tx, lambda_):
    """Ridge sur labels {-1,+1}. Retourne w."""
    w, _ = ridge_regression(y_pm1, tx, lambda_)
    return w


def fit_logistic(y01, tx, lambda_, gamma, max_iters, pos_weight):
    w0 = np.zeros(tx.shape[1])
    w, _ = weighted_reg_logistic(y01, tx, lambda_, w0, max_iters, gamma, pos_weight)
    return w


def score_linear(tx, w):
    return tx @ w


def score_logistic(tx, w):
    return stable_sigmoid(tx @ w)

# --------------------------------------------------------------------------
# Validation croisée
# --------------------------------------------------------------------------

def build_k_indices(n, k_fold, seed=1):
    """Découpe aléatoire des indices en k_fold blocs."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k_fold)


def cross_validate(fit_fn, score_fn, y_pm1, tx, k_fold=5, seed=1):
    """CV k-fold. Retourne (F1 optimal sur scores out-of-fold, seuil, metrics).

    fit_fn(y_pm1_train, tx_train) -> w
    score_fn(tx_val, w)           -> scores continus

    On concatène les scores out-of-fold puis on choisit UN seuil global :
    plus stable que d'optimiser le seuil fold par fold.
    """
    folds = build_k_indices(len(y_pm1), k_fold, seed)
    oof = np.zeros(len(y_pm1))
    for k in range(k_fold):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(k_fold) if j != k])
        w = fit_fn(y_pm1[train_idx], tx[train_idx])
        oof[val_idx] = score_fn(tx[val_idx], w)
    y01 = to_01(y_pm1).astype(int)
    thr, f1 = best_threshold(oof, y01)
    return f1, thr, metrics(y01, (oof >= thr).astype(int))


# --------------------------------------------------------------------------
# Recherche d'hyperparamètres
# --------------------------------------------------------------------------


def tune_ridge(y_pm1, tx, lambdas, k_fold=5, seed=1):
    results = []
    for lam in lambdas:
        f1, thr, m = cross_validate(
            lambda y, x: fit_ridge(y, x, lam), score_linear, y_pm1, tx, k_fold, seed
        )
        print(f"[ridge] lambda={lam:.1e}  F1={f1:.4f}  rec={m['rec']:.3f}  prec={m['prec']:.3f}")
        results.append({"lambda_": lam, "f1": f1, "thr": thr})
    return max(results, key=lambda r: r["f1"]), results


def tune_logistic(y_pm1, tx, lambdas, gammas, pos_weights, max_iters=300, k_fold=5, seed=1):
    results = []
    for lam in lambdas:
        for gam in gammas:
            for pw in pos_weights:
                f1, thr, m = cross_validate(
                    lambda y, x: fit_logistic(to_01(y), x, lam, gam, max_iters, pw),
                    score_logistic, y_pm1, tx, k_fold, seed,
                )
                print(f"[logreg] lambda={lam:.1e} gamma={gam} pw={pw}  F1={f1:.4f}")
                results.append(
                    {"lambda_": lam, "gamma": gam, "pos_weight": pw, "f1": f1, "thr": thr}
                )
    return max(results, key=lambda r: r["f1"]), results


# --------------------------------------------------------------------------
# Entraînement final + prédiction
# --------------------------------------------------------------------------


def train_and_predict(y_pm1, tx_train, tx_test, best, model="logistic", max_iters=300):
    """Réentraîne sur TOUT le train avec les meilleurs hyperparamètres,
    applique le seuil trouvé en CV, et renvoie des prédictions en {-1,+1}."""
    if model == "ridge":
        w = fit_ridge(y_pm1, tx_train, best["lambda_"])
        scores = score_linear(tx_test, w)
    else:
        w = fit_logistic(
            to_01(y_pm1), tx_train, best["lambda_"], best["gamma"], max_iters, best["pos_weight"]
        )
        scores = score_logistic(tx_test, w)
    y01 = (scores >= best["thr"]).astype(int)
    return to_pm1(y01), w


if __name__ == "__main__":
    # Exemple d'utilisation une fois le preprocessing prêt :
    #
    # from preprocessing import preprocess          # <- le travail de tes amies
    # from helpers import load_csv_data, create_csv_submission
    # x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")
    # tx_train, tx_test = preprocess(x_train, x_test)
    #
    # best_ridge, _ = tune_ridge(y_train, tx_train, np.logspace(-8, -2, 7))
    # best_log, _ = tune_logistic(
    #     y_train, tx_train,
    #     lambdas=[1e-6, 1e-4], gammas=[0.5, 1.0], pos_weights=[1, 3, 6],
    # )
    # y_pred, w = train_and_predict(y_train, tx_train, tx_test, best_log)
    # create_csv_submission(test_ids, y_pred, "submission.csv")
    pass