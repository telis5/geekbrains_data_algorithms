from linear.regression import LinearRegression
from .core import iterate


class GradientDescent(LinearRegression):
    def _iterate(self, X: list[list[float]], y: list[float], s: float):
        self._w: list[float] = iterate(X, y, self._w, s)

    from .fit import fit
