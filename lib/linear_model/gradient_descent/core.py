import numpy as np


def quality(X: np.array, y: np.array, w: np.array) -> float:
    return sum((X.dot(w) - y)**2) / X[0].shape[0]


def quality_gradient(X: np.array, y: np.array, w: np.array) -> np.array:
    return 2 * np.dot(X.transpose(), X.dot(w) - y) / X.shape[0]


def initialize(X: np.array) -> np.array:
    return np.zeros(shape=X.shape[1])


def iterate(X: np.array, y: np.array, w: np.array, s: float) -> np.array:
    return w - s * quality_gradient(X, y, w)


def residual(w: tuple[np.array, np.array]) -> float:
    return sum((w[1] - w[0])**2)**.5


def stop(r: float, r_min: float, r_max: float, i: int, i_max: int) -> bool:
    return r <= r_min or r >= r_max or i >= i_max


def error(X: np.array, y: np.array, w: np.array) -> float:
    return quality(X, y, w)


def prepare(X: np.array, y: np.array, w: np.array, *args) -> tuple:
    return *args, error(X, y, w)
