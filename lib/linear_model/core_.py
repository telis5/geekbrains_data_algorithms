def _expand(X: list[list[float]]) -> list[list[float]]:
    return [xi.insert(0, 1) for xi in X]


def _predict(X: list[list[float]], w: list) -> list:
    return [sum([xij * wj for xij, wj in zip(xi, w)]) for xi in X]
