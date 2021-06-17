import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

from example_1_knn import KNN
from example_1_makegrid import makegrid
from example_1_relation import euclidean_distance, accuracy
from example_1_plot import scatter_1, scatter_2, plot_1, pcolormesh_1

# %% Set parameters.

neighbor_numbers = list(range(1, 5, 1)) + \
    list(range(5, 30, 5)) + \
    list(range(30, 90, 10))

dataset_parameters = {'return_X_y': True}

split_parameters = {
    'test_size': .2,
    'random_state': 1
}

grid_size = (100, 100)

colormap_parameters = {
    'normal': {'colors': ['red', 'green', 'blue']},
    'light': {'colors': ['#FFAAAA', '#AAFFAA', '#00AAFF']}
}

# %% Configure parameters.

colormaps = {
    key: ListedColormap(**parameters)
    for key, parameters in colormap_parameters.items()
}

# %% Lead dataset.

X, y = load_iris(**dataset_parameters)
X = X[:, :2]

# %% Plot dataset.

subplots_parameters = {'figsize': (8, 8)}

_, axis = plt.subplots(**subplots_parameters)

scatter_1(
    X, y,
    indexes=(0, 1),
    colormap=colormaps['normal'],
    axis=axis
)

plt.show()

# %% Split dataset.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    **split_parameters
)

# %% Plot split.

subplots_parameters = {'figsize': (8, 8)}

_, axis = plt.subplots(**subplots_parameters)

scatter_2(
    (X_train, X_test),
    (y_train, y_test),
    indexes=(0, 1),
    colormaps=[colormaps['light'], colormaps['normal']],
    axis=axis
)

plt.show()

# %% Make grid.

X_grid = makegrid(
    data=X,
    size=grid_size
)


# %% Make predictions.

model = KNN()
model.fit(X_train, y_train, metric=euclidean_distance)

a_test, a_grid = [
    [
        model.predict(X, k)
        for k in neighbor_numbers
    ]
    for X in (X_test, X_grid)
]

# %% Calculate metrics.

test_metrics = [accuracy(ai, y_test) for ai in a_test]

# %% Plot metrics.

subplots_parameters = {'figsize': (8, 8)}
axis_parameters = {
    'xlabel': 'k',
    'ylabel': 'accuracy'
}

_, axis = plt.subplots(**subplots_parameters)

plot_1(
    neighbor_numbers, test_metrics,
    axis_parameters=axis_parameters,
    axis=axis
)

plt.show()

# %% Plot hypersurfaces.

k = 5
i = neighbor_numbers.index(k)

subplots_parameters = {'figsize': (8, 8)}
figure_parameters = {'suptitle': f'accuracy: {test_metrics[i]:.3g}'}

figure, axis = plt.subplots(**subplots_parameters)

pcolormesh_1(
    X_grid=X_grid,
    y_grid=np.array(a_grid[i]),
    grid_size=grid_size,
    X=X_test,
    y=y_test,
    indexes=(0, 1),
    colormaps=[colormaps['light'], colormaps['normal']],
    axis=axis
)

figure.suptitle(figure_parameters['suptitle'])

plt.show()
