import numpy as np


def makegrid(
    data: np.array,
    size: tuple[int, int] = (100, 100)
) -> np.array:
    n_x, n_y = size
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_x),
        np.linspace(y_min, y_max, n_y)
    )

    return np.c_[xx.ravel(), yy.ravel()]
