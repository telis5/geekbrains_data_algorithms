from linear_model import LinearModel

from .core import quality, estimate, fit, error


class LinearRegression(LinearModel):
    def quality(self, X: list[list[float]], y: list[float]) -> float:
        return quality(self._expand(X), y, self._w)

    def estimate(self, X: list[list[float]], y: list[float]):
        self._w: list = estimate(self._expand(X), y)

    def fit(self, X: list[list[float]], y: list[float]):
        self._w: list = fit(self._expand(X), y)

    def error(self, X: list[list[float]], y: list[float]) -> float:
        return error(self._expand(X), y, self._w)
