import numpy as np
from gradient_descent import GradientDescent


class LassoGradientDescent(GradientDescent):
    def __init__(self, a: float = 1, intercept: bool = True):
        self._a = a
        self._intercept = intercept

    @staticmethod
    def _quality(X: np.array, y: np.array, w: np.array, a: float) -> float:
        return sum((X.dot(w) - y)**2) / X[0].shape[0] + a * sum(abs(w))

    def quality(self, X: np.array, y: np.array) -> float:
        return self._quality(self._extended(X), y, self._w, self._a)

    @staticmethod
    def _quality_gradient(
            X: np.array, y: np.array, w: np.array, a: float
    ) -> np.array:
        return 2 * np.dot(X.transpose(), X.dot(w) - y) / X.shape[0] + \
            a * np.sign(w)

    @classmethod
    def _iterate(
        cls, X: np.array, y: np.array, w: np.array, s: float, a: float
    ) -> np.array:
        return w - s * cls._quality_gradient(X, y, w, a)

    def iterate(self, X: np.array, y: np.array, s: float) -> np.array:
        return self._iterate(X, y, self._w, s, self._a)
