import numpy as np


def uniform_split(
    data: np.array, size: tuple[int, int] = (100, 100)
) -> np.array:
    n_x, n_y = size
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_x),
        np.linspace(y_min, y_max, n_y)
    )

    return np.c_[xx.ravel(), yy.ravel()]


def error_matrix(a: list, y: list) -> tuple[tuple[int, int], tuple[int, int]]:
    TP = FP = TN = FN = 0
    for ai, yi in zip(a, y):
        if ai:
            if ai == yi:
                TP += 1
            else:
                FP += 1
        else:
            if ai == yi:
                TN += 1
            else:
                FN += 1
    return ((TP, FP), (FN, TN))


def error_metrics(
    e: tuple[tuple[int, int], tuple[int, int]]
) -> tuple[float, float, float, float]:
    ((TP, FP), (FN, TN)) = e
    a = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else None
    p = TP / (TP + FP) if TP + FP else None
    r = TP / (TP + FN) if TP + FN else None
    f = 2 * p * r / (p + r) \
        if p is not None and r is not None and p + r else None
    return a, p, r, f
