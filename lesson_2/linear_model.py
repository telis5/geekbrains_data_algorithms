import numpy as np


class LinearModel:
    def __init__(self, intercept: bool = True):
        self._intercept = intercept

    # @staticmethod
    # def _l1(x: list[float]) -> float:
    #     return sum([abs(xi) for xi in x])

    @staticmethod
    def _l1(x: np.array) -> float:
        return sum(abs(x))

    # @staticmethod
    # def _l2(x: list[float]) -> float:
    #     return sum([xi**2 for xi in x])**.5

    @staticmethod
    def _l2(x: np.array) -> float:
        return sum(x**2)**.5

    # @staticmethod
    # def _mae(X: tuple[list[float], list[float]]) -> float:
    #     return sum([abs(x1i - x0i) for x0i, x1i in zip(*X)]) / len(X[0])

    @staticmethod
    def _mae(X: tuple[np.array, np.array]) -> float:
        return sum(abs(X[1] - X[0])) / X[0].shape[0]

    # @staticmethod
    # def _mse(X: tuple[list[float], list[float]]) -> float:
    #     return sum([(x1i - x0i)**2 for x0i, x1i in zip(*X)]) / len(X[0])

    @staticmethod
    def _mse(X: tuple[np.array, np.array]) -> float:
        return sum((X[1] - X[0])**2) / X[0].shape[0]

    @property
    def intercept(self) -> float:
        return self._w[0] if self._intercept else 0

    @property
    def coef(self) -> list[float]:
        return self._w[1:] if self._intercept else self._w

    # @staticmethod
    # def _extend(X: list[list[float]]) -> list[list[float]]:
    #     return [xi.insert(0, 1) for xi in X]

    @staticmethod
    def _extend(X: np.array) -> np.array:
        return np.hstack((np.ones(shape=(X.shape[0], 1)), X.copy()))

    def _extended(self, X: np.array) -> np.array:
        return self._extend(X) if self._intercept else X

    # @staticmethod
    # def _predict_(X: list, w: list) -> list:
    #     return [sum([xij * wj for xij, wj in zip(xi, w)]) for xi in X]

    @staticmethod
    def _predict(X: np.array, w: np.array) -> np.array:
        return X.dot(w)

    def predict(self, X: np.array) -> np.array:
        return self._predict(self._extended(X), self._w)
