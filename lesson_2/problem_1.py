from gradient_descent import GradientDescent
from stochastic_gradient_descent import StochasticGradientDescent


def prepare(
        self, X: list[list[float]], y: list[float], i: int, r: float
) -> tuple:
    return (i, r, self._error(X, y, self._w))


def save(save_to, *args):
    if isinstance(save_to, list):
        save_to.append(*args)


class GradientDescent1(GradientDescent):
    def _prepare(self, *args, **kwargs):
        return prepare(self, *args, **kwargs)

    _save = staticmethod(save)


class StochasticGradientDescent1(StochasticGradientDescent):
    def _prepare(self, *args, **kwargs):
        return prepare(self, *args, **kwargs)

    _save = staticmethod(save)
