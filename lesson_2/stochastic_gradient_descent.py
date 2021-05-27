import numpy as np
from gradient_descent import GradientDescent


class StochasticGradientDescent(GradientDescent):
    @staticmethod
    def _quality_gradient(X: np.array, y: np.array, w: np.array) -> np.array:
        i = np.random.randint(X.shape[0])
        # return 2 * X.T[:, i] * (X[i].dot(w) - y[i]) / X.shape[0]
        return 2 * X[i] * (X[i].dot(w) - y[i]) / X.shape[0]
