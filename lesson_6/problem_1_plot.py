# import numpy as np
import matplotlib.pyplot as plt


def plot_1(
    metric_index,
    tree_indexes,
    depth_indexes,
    metrics,
    number_trees,
    max_depths
):
    dataset_labels = ['train', 'test']
    metric_labels = ['MSE', 'MAE', 'STD', r'${R}^{2}$']

    subplots_params = {'figsize': (8, 8)}
    axes_params = {
        'xlabel': 'number of trees',
        'ylabel': metric_labels[metric_index]
    }
    legend_params = {'title': 'depth: dataset'}

    figure, axes = plt.subplots(**subplots_params)

    for j in depth_indexes:
        for i in range(metrics.shape[0]):
            plot_data = [
                number_trees[tree_indexes],
                metrics[i, j, tree_indexes, metric_index]
            ]
            plot_params = {
                'label': '{}: {}'.format(max_depths[j], dataset_labels[i])
            }
            axes.plot(
                *plot_data,
                **plot_params
            )

    axes.set(**axes_params)
    axes.legend(**legend_params)
    axes.grid()
    plt.show()


def plot_2(
    metric_index,
    tree_indexes,
    depth_indexes,
    metrics,
    number_trees,
    max_depths
):
    dataset_labels = ['train', 'test']
    metric_labels = ['MSE', 'MAE', 'STD', r'${R}^{2}$']

    subplots_params = {'figsize': (8, 8)}
    axes_params = {
        'xlabel': 'maximum depth',
        'ylabel': metric_labels[metric_index],
        'xticks': max_depths[depth_indexes]
    }
    plt.fmt = 'o--'
    legend_params = {'title': 'trees: dataset'}

    figure, axes = plt.subplots(**subplots_params)

    for j in tree_indexes:
        for i in range(metrics.shape[0]):
            plot_data = [
                max_depths[depth_indexes],
                metrics[i, depth_indexes, j, metric_index]
            ]
            plot_params = {
                'label': '{}: {}'.format(number_trees[j], dataset_labels[i])
            }
            axes.plot(
                *plot_data,
                plt.fmt,
                **plot_params
            )

    axes.set(**axes_params)
    axes.legend(**legend_params)
    axes.grid()
    plt.show()
