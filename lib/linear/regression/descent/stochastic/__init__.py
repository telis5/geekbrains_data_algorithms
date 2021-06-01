from linear.regression.descent import GradientDescent
from .core import iterate


class StochasticGradientDescent(GradientDescent):
    def _iterate(self, X: list[list[float]], y: list[float], s: float):
        self._w: list[float] = iterate(X, y, self._w, s)
