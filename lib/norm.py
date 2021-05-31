import numpy as np


def l1(x: np.array) -> float:
    return sum(abs(x))


def l2(x: np.array) -> float:
    return sum(x**2)**.5
