import numpy as np


def calculate_mse(err):
    """Calculate the mse for vector err.
    Args:
        e: numpy array of shape (N,), N is the number of samples.
    returns: 
        mse: scalar    
    """
    return 1 / 2 * np.mean(err**2)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        w = w - gamma * grad

        ws.append(w)
        losses.append(loss)
        
    return losses[-1], ws[-1]

  
def mean_squared_error_sgd (y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, err = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = calculate_mse(err)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            
    return losses[-1], ws[-1]


def compute_loss(y, tx, w):
    """Calculate the mse loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape(D,)

    Returns:
        Scalar

    """
    e = y - tx.dot(w)
    return calculate_mse(e) 
            
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.
    
        Args:
            y: numpy array of shape (N, ), target values.
            tx: numpy array of shape (N, D), feature matrix.
            lambda_: regularization parameter.
    
        Returns:
            w: optimal weights, numpy array of shape (D, ).
            loss: mean squared error without the regularization penalty.
        """
    N = len(y)
    lambda_prime = 2 * N * lambda_

    a = tx.T.dot(tx) + lambda_prime * np.eye(tx.shape[1])
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    
    err = y - tx.dot(w)
    loss = calculate_mse(err)

    return w, loss

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return (1 + np.exp(-t))**(-1)

def compute_gradient_logistic (y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred-y) * (1/y.shape[0])

    return grad

def  compute_logistic_loss (y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    
    return np.squeeze(-loss).item() * (1 / y.shape[0])



def logistic_regression (y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Returns:
        w: the last weight vector
        loss: the corresponding loss (negative log-likelihood, averaged over N)
    """
    
    w = initial_w
    losses = []
    ws = [initial_w]
    for _ in range(max_iters):
        grad = compute_gradient_logistic (y, tx, w)
        w -= gamma * grad
        loss = compute_logistic_loss(y, tx, w)
        losses.append(loss)
        ws.append(w)

    return ws[-1], losses[-1]

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent (y in {0, 1}) and a regularization term.

    Returns:
        w: the last weight vector
        loss: the corresponding loss (negative log-likelihood, averaged over N)
    """
    
    w = initial_w
    losses = []
    ws = [initial_w]
    for _ in range(max_iters):
        grad = compute_gradient_logistic (y, tx, w) + 2 * lambda_ * w
        w -= gamma * grad
        loss = compute_logistic_loss(y, tx, w)
        losses.append(loss)
        ws.append(w)
    
    return ws[-1], losses[-1]