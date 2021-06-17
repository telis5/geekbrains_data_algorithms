from numpy import array
from matplotlib.axes import Axes


def scatter_1(
    X: array,
    y: array,
    axis: Axes,
    axis_parameters: dict,
    scatter_parameters: dict,
    legend_parameters: dict
):
    scatter_data = {
        'x': X[:, 0],
        'y': X[:, 1],
        'c': y
    }
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
    axis: Axes,
    axis_parameters: dict,
    scatter_parameters: list[dict],
    legend_parameters: dict
):
    for i in range(len(X_)):
        scatter_data = {
            'x': X_[i][:, 0],
            'y': X_[i][:, 1],
            'c': y_[i]
        }
        scatter = axis.scatter(
            **scatter_data,
            **scatter_parameters[i]
        )
    axis.legend(
        *scatter.legend_elements(),
        **legend_parameters
    )
    axis.set(**axis_parameters)


def plot_1(
    x: list,
    y: list[list],
    axis: Axes,
    axis_parameters: dict,
    plot_fmt: str,
    plot_parameters: list[dict],
    legend_parameters: dict
):
    for i in range(len(y)):
        plot_data = (x, y[i])
        axis.plot(
            *plot_data,
            plot_fmt,
            **plot_parameters[i]
        )
    axis.set(**axis_parameters)
    axis.legend(**legend_parameters)
