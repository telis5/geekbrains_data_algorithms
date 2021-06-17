from numpy import array
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes


def scatter_1(
    X: array,
    y: array,
    indexes: tuple[int],
    colormap: ListedColormap,
    axis: Axes
):
    axis_parameters = {
        'xlabel': f'index {indexes[0]}',
        'ylabel': f'index {indexes[1]}'
    }
    legend_parameters = {'title': 'classes'}

    scatter_data = {
        'x': X[:, indexes[0]],
        'y': X[:, indexes[1]],
        'c': y
    }
    scatter_parameters = {'cmap': colormap}

    scatter = axis.scatter(
        **scatter_data,
        **scatter_parameters
    )
    axis.legend(
        *scatter.legend_elements(),
        **legend_parameters
    )
    axis.set(**axis_parameters)


def scatter_2(
    X_: list[array],
    y_: list[array],
    indexes: tuple[int],
    colormaps: list[ListedColormap],
    axis: Axes
):
    axis_parameters = {
        'xlabel': f'index {indexes[0]}',
        'ylabel': f'index {indexes[1]}'
    }
    legend_parameters = {'title': 'classes'}

    for i in range(len(X_)):
        scatter_data = {
            'x': X_[i][:, indexes[0]],
            'y': X_[i][:, indexes[1]],
            'c': y_[i]
        }
        scatter_parameters = {'cmap': colormaps[i]}
        scatter = axis.scatter(
            **scatter_data,
            **scatter_parameters
        )
    axis.legend(
        *scatter.legend_elements(),
        **legend_parameters
    )
    axis.set(**axis_parameters)


def plot_1(
    x: list,
    y: list,
    axis_parameters: dict,
    axis: Axes
):
    axis_parameters_ = axis_parameters.copy()
    axis_parameters_['xticks'] = x

    plot_data = [x, y]
    plot_parameters = ['o--']

    axis.plot(
        *plot_data,
        *plot_parameters
    )
    axis.set(**axis_parameters_)


def pcolormesh_1(
    X_grid: array, y_grid: array,
    grid_size: tuple[int, int],
    X: array, y: array,
    indexes: tuple[int],
    colormaps: list[ListedColormap],
    axis: Axes
):
    axis_parameters = {
        'xlabel': f'index {indexes[0]}',
        'ylabel': f'index {indexes[1]}'
    }
    legend_parameters = {'title': 'classes'}

    pcolormesh_data = [
        X_grid[:, 0].reshape(grid_size),
        X_grid[:, 1].reshape(grid_size),
        y_grid.reshape(grid_size)
    ]
    pcolormesh_parameters = {'cmap': colormaps[0]}

    axis.pcolormesh(
        *pcolormesh_data,
        **pcolormesh_parameters
    )

    scatter_data = {
        'x': X[:, indexes[0]],
        'y': X[:, indexes[1]],
        'c': y
    }
    scatter_parameters = {'cmap': colormaps[1]}

    scatter = axis.scatter(
        **scatter_data,
        **scatter_parameters
    )
    axis.legend(
        *scatter.legend_elements(),
        **legend_parameters
    )

    axis.set(**axis_parameters)
