import numpy as np

from os import path
from json import loads, dumps
from sklearn.datasets import make_regression
from linear_model import GradientDescentRegressor

# %% Set parameters.

parameters_name = 'example_1_par.json'
cases_name = 'example_1_log.json'

directory_name = path.dirname(path.realpath(__file__))

with open(path.join(directory_name, parameters_name), 'r') as file:
    parameters = loads(file.read())

step_sizes = np.arange(**parameters['regressor_fit']['step_size'])
step_sizes.sort()
del parameters['regressor_fit']['step_size']

residual_mins = np.logspace(**parameters['regressor_fit']['residual_min'])
residual_mins.sort()
del parameters['regressor_fit']['residual_min']

log_names = [
    parameters['regressor_fit']['log_name'].format(_)
    for _ in range(1, len(step_sizes) + 1)
]
del parameters['regressor_fit']['log_name']

cases = {
     step_size: log_name
     for step_size, log_name in zip(step_sizes, log_names)
}

# %% Make dataset.

X, y = make_regression(**parameters['dataset'])

# %% Fit cases.

regressor = GradientDescentRegressor()

for step_size, log_name in cases.items():
    print(log_name)
    with open(path.join(directory_name, log_name), 'w') as log_file:
        regressor.fit(
            samples=X,
            targets=y,
            step_size=step_size,
            residual_min=residual_mins.min(),
            log_file=log_file,
            **parameters['regressor_fit']
        )

# %% Save info.

with open(path.join(directory_name, cases_name), 'w') as file:
    file.write(dumps(cases))
