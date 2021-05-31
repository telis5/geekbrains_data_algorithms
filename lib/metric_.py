def mae(X: tuple[list[float], list[float]]) -> float:
    return sum([abs(x1i - x0i) for x0i, x1i in zip(*X)]) / len(X[0])


def mse(X: tuple[list[float], list[float]]) -> float:
    return sum([(x1i - x0i)**2 for x0i, x1i in zip(*X)]) / len(X[0])
