from linear_model import LinearModel
from .core import quality, iterate, error


class GradientDescent(LinearModel):
    def quality(self, X: list[list[float]], y: list[float]) -> float:
        return quality(self._expand(X), y, self._w)

    def _iterate(self, X: list[list[float]], y: list[float], s: float):
        self._w: list[float] = iterate(X, y, self._w, s)

    def error(self,  X: list[list[float]], y: list[float]) -> float:
        return error(self._expand(X), y, self._w)

    from .fit import fit
