from linear.regression.descent import GradientDescent
from .core import quality, iterate


class RidgeGradientDescent(GradientDescent):
    def __init__(self, a: float = 1, intercept: bool = True):
        self._a = a
        self._intercept = intercept

    def quality(self, X: list[list[float]], y: list[float]) -> float:
        return quality(self._expand(X), y, self._w)

    def _iterate(self, X: list[list[float]], y: list[float], s: float):
        self._w: list[float] = iterate(X, y, self._w, s, self._a)
