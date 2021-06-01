import numpy as np


def expand(X: np.array) -> np.array:
    return np.hstack((np.ones(shape=(X.shape[0], 1)), X))
