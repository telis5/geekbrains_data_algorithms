def mse(a: list[float], y: list[float]) -> float:
    return sum([(ai - yi) ** 2 for ai, yi in zip(a, y)]) / len(a)


def mae(a: list[float], y: list[float]) -> float:
    return sum([abs(ai - yi)for ai, yi in zip(a, y)]) / len(a)


def std(a: list[float], y: list[float]) -> float:
    return sum([(ai - yi) ** 2 for ai, yi in zip(a, y)]) ** .5 / len(a)


def r2(a: list[float], y: list[float]) -> float:
    y_mean = sum(y) / len(y)
    # return 1 - \
    #     sum([(yi - ai) ** 2 for yi, ai in zip(y, a)]) / \
    #     sum([(yi - y_mean) ** 2 for yi in y])
    return sum([(ai - y_mean) ** 2 for ai in a]) / \
        sum([(yi - y_mean) ** 2 for yi in y])
