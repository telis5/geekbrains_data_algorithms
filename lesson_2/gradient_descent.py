import numpy as np
from linear_regression import LinearRegression


class GradientDescent(LinearRegression):
    @staticmethod
    def _initialize(X: np.array) -> np.array:
        return np.zeros(shape=X.shape[1])

    @staticmethod
    def _loss_gradient(X: np.array, y: np.array, w: np.array) -> np.array:
        return 2 * np.dot(X.transpose(), X.dot(w) - y) / X.shape[0]

    @classmethod
    def _iterate(
        cls, X: np.array, y: np.array, w: np.array, s: float
    ) -> tuple:
        return w - s * cls._loss_gradient(X, y, w)

    # @classmethod
    # def _residual(cls, W: tuple[list[float], list[float]]) -> float:
    #     return cls._l2([w1i - w0i for w0i, w1i in zip(*W)])

    @staticmethod
    def _residual(w: tuple[np.array, np.array]) -> float:
        return sum((w[1] - w[0])**2)**.5

    @staticmethod
    def _stop(
        r: float, r_min: float, r_max: float, i: int, i_max: int
    ) -> bool:
        return r <= r_min or r >= r_max or i >= i_max

    def fit(
        self,
        X: list[list[float]],
        y: list[float],
        s: float = 1e-1,
        w0: list[float] = None,
        r_min: float = 1e-6,
        r_max: float = 1e6,
        i_max: int = 1e3,
        copy: bool = True,
        save_to=None
    ) -> int:
        if self._intercept:
            X_ = self._extend(X)
        elif copy:
            X_ = X.copy()
        else:
            X_ = X

        if w0 is None:
            self._w = self._initialize(X_)
        elif copy:
            self._w = w0.copy()
        else:
            self._w = w0

        i, r = 0, None
        if save_to is not None:
            self._save(save_to, self._prepare(X_, y, i, r))

        while True:
            self._w, w_ = self._iterate(X_, y, self._w, s), self._w.copy()
            r = self._residual((w_, self._w))
            i += 1

            if save_to is not None:
                self._save(save_to, self._prepare(X_, y, i, r))

            if self._stop(r, r_min, r_max, i, i_max):
                break

        return i
