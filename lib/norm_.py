def l1(x: list[float]) -> float:
    return sum([abs(xi) for xi in x])


def l2(x: list[float]) -> float:
    return sum([xi**2 for xi in x])**.5
