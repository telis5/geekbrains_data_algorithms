import numpy as np


def mae(X: tuple[np.array, np.array]) -> float:
    return sum(abs(X[1] - X[0])) / X[0].shape[0]


def mse(X: tuple[np.array, np.array]) -> float:
    return sum((X[1] - X[0])**2) / X[0].shape[0]
