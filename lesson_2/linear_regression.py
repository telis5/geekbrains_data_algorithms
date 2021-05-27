import numpy as np
from linear_model import LinearModel


class LinearRegression(LinearModel):
    # @classmethod
    # def _loss(
    #     cls, X: list[list[float]], y: list[float], w: list[float]
    # ) -> float:
    #     return cls._mse((cls._predict(X, w), y))

    @staticmethod
    def _loss(X: np.array, y: np.array, w: np.array) -> float:
        return sum((X.dot(w) - y)**2) / X[0].shape[0]

    def loss(self, X: np.array, y: np.array) -> float:
        return self._loss(self._extended(X), y, self._w)

    # @classmethod
    # def _error(
    #     cls, X: list[list[float]], y: list[float], w: list[float]
    # ) -> float:
    #     return cls._mse((cls._predict(X, w), y))

    @staticmethod
    def _error(X: np.array, y: np.array, w: np.array) -> float:
        return sum((X.dot(w) - y)**2) / X[0].shape[0]

    def error(self, X: np.array, y: np.array) -> float:
        return self._error(self._extended(X), y, self._w)

    def _estimate(X: np.array, y: np.array) -> np.array:
        return np.dot(
            np.linalg.inv(np.dot(X.transpose(), X)),
            np.dot(X.transpose(), y)
        )

    def estimate(self, X: np.array, y: np.array):
        self._w = self._fit(self._extended(X), y)

    @staticmethod
    def _fit(X: np.array, y: np.array) -> np.array:
        return np.linalg.solve(
            np.dot(X.transpose(), X),
            np.dot((X.transpose()), y)
        )

    def fit(self, X: np.array, y: np.array):
        self._w = self._fit(self._extended(X), y)
