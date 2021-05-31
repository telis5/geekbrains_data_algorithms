import numpy as np


def quality(X: np.array, y: np.array, w: np.array) -> float:
    return sum((X.dot(w) - y)**2) / X[0].shape[0]


def estimate(X: np.array, y: np.array) -> np.array:
    return np.dot(
        np.linalg.inv(np.dot(X.transpose(), X)),
        np.dot(X.transpose(), y)
    )


def fit(X: np.array, y: np.array) -> np.array:
    return np.linalg.solve(
        np.dot(X.transpose(), X),
        np.dot((X.transpose()), y)
    )


def error(X: np.array, y: np.array, w: np.array) -> float:
    return quality(X, y, w)
