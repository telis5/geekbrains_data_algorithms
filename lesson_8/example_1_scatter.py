import numpy as np
import matplotlib.pyplot as plt


def scatter(
    X: np.array, y: np.array,
    ncols: int = 1, figsize: tuple[int, int] = (8, 8)
):
    n = X.shape[-1]
    naxes = int(n * (n - 1) / 2)
    nrows = naxes // ncols if naxes > 1 else 1

    figure, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows) if n > 1 else figsize,
        constrained_layout=True
    )

    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            axis = axes.flat[k] if n > 2 else axes
            scatter = axis.scatter(x=X[:, i], y=X[:, j], c=y)
            axis.set(xlabel=i, ylabel=j)
            axis.legend(*scatter.legend_elements())
            k += 1

    plt.show()
