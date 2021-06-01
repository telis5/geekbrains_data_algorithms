from abc import ABC
from .core import expand


class LinearModel(ABC):
    def __init__(self, intercept: bool = True):
        self._intercept = intercept
        self._w: list[float]

    @property
    def intercept(self) -> float:
        return self._w[0] if self._intercept else 0

    @property
    def coef(self) -> list[float]:
        return self._w[1:] if self._intercept else self._w

    def _expand(self, X: list[list[float]]) -> list[list[float]]:
        return expand(X) if self._intercept else X
