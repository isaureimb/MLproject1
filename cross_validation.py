import numpy as np

def build_k_indices(y, k_fold, seed=1):
    """Split the sample indices into k folds, keeping the class proportions.
 
    Args:
        y: numpy array of shape (N,), labels in {-1, 1}
        k_fold: number of folds
        seed: random seed, for reproducibility
 
    Returns:
        list of k numpy arrays, each containing the indices of one fold
    """
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_fold, lambda_, degree, seed, model_train, loss_function):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527596753, 0.33555914361295197)
    """

    k_indices = build_k_indices(y, k_fold, seed)
    rmse_te = []
    
    for k in range(k_fold):
        te_indice = k_indices[k]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indice = tr_indice.reshape(-1)
        y_te = y[te_indice]
        y_tr = y[tr_indice]
        x_te = x[te_indice]
        x_tr = x[tr_indice]

        w, _ = model_train(y_tr, x_tr)
        loss_te = loss_function(y_te, x_te, w)
        rmse_te.append(loss_te)
   
    return np.mean(rmse_te)