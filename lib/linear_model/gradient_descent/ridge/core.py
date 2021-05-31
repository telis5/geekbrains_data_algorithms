import numpy as np


def quality(X: np.array, y: np.array, w: np.array, a: float) -> float:
    return sum((X.dot(w) - y)**2) / X[0].shape[0] + a * sum(w**2)


def quality_gradient(
        X: np.array, y: np.array, w: np.array, a: float
) -> np.array:
    return 2 * np.dot(X.transpose(), X.dot(w) - y) / X.shape[0] + 2 * a * w


def iterate(
    X: np.array, y: np.array, w: np.array, s: float, a: float
) -> np.array:
    return w - s * quality_gradient(X, y, w, a)
