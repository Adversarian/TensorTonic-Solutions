import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X_arr = np.asarray(X); y_arr = np.asarray(y)
    W = np.zeros(X.shape[1]); b = 0.0
    n = X.shape[0]
    # GD
    for _ in range(steps):
        pred = _sigmoid(X@W + b)
        err = pred - y
        W -= lr * (X.T @ err / n)
        b -= lr * np.mean(err)
    return W, b