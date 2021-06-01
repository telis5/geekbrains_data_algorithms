from linear.core import expand
from .core import initialize, residual, stop, prepare
from .save import save


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
        X_ = expand(X)
    elif copy:
        X_ = X.copy()
    else:
        X_ = X

    if w0 is None:
        self._w = initialize(X_)
    elif copy:
        self._w = w0.copy()
    else:
        self._w = w0

    i, r = 0, None
    if save_to is not None:
        save(save_to, prepare(X_, y, self._w, i, r))

    while True:
        w_ = self._w.copy()
        self._iterate(X_, y, s)
        r = residual((w_, self._w))
        i += 1

        if save_to is not None:
            save(save_to, prepare(X_, y, self._w, i, r))

        if stop(r, r_min, r_max, i, i_max):
            break

    return i
