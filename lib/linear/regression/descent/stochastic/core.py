import numpy as np


def stochastic_quality_gradient(
    X: np.array, y: np.array, w: np.array
) -> np.array:
    i = np.random.randint(X.shape[0])
    return 2 * X[i] * (X[i].dot(w) - y[i]) / X.shape[0]


def iterate(X: np.array, y: np.array, w: np.array, s: float) -> np.array:
    return w - s * stochastic_quality_gradient(X, y, w)
