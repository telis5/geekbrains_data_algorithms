from linear import LinearModel
from .core import quality, fit, predict, error


class LinearRegression(LinearModel):
    def __init__(self, intercept: bool = True):
        self._intercept = intercept
        self._w: list[float]

    def quality(self, X: list[list[float]], y: list[float]) -> float:
        return quality(self._expand(X), y, self._w)

    def fit(self, X: list[list[float]], y: list[float]):
        self._w: list = fit(self._expand(X), y)

    def predict(self, X: list[float]) -> list[float]:
        return predict(self._expand(X), self._w)

    def error(self, X: list[list[float]], y: list[float]) -> float:
        return error(self._expand(X), y, self._w)
