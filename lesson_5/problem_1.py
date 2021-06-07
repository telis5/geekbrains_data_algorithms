# import numpy as np


# def map_to_grid(
#     func: object, data: np.array, shape: tuple[int, int]
# ) -> tuple[np.array, np.array]:
#     x_min, x_max = data[:, 0].min(), data[:, 0].max()
#     y_min, y_max = data[:, 1].min(), data[:, 1].max()

#     xx, yy = np.meshgrid(
#         np.linspace(x_min, x_max, shape[0]),
#         np.linspace(y_min, y_max, shape[1])
#     )

#     zz = func(np.c_[xx.ravel(), yy.ravel()])
#     zz = np.array(zz).reshape(shape)

#     return xx, yy, zz


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
