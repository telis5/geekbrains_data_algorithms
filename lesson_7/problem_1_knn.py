def euclidean_distance(x: list[float], y: list[float]) -> float:
    return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)]) ** .5


def predict(
    X: list[list],
    X0: list[list],
    y0: list,
    k: int,
    weighted: int,
    distance: object,
    weight: object
) -> list:
    a = []
    for xi in X:
        di = [(distance(xi, x0j), y0j) for x0j, y0j in zip(X0, y0)]
        ci = {_: 0 for _ in set(y0)}

        if weighted:
            if weighted == 1:
                if weight is None:
                    def _weight(i):
                        return (k + 1 - i) / k
                else:
                    _weight = weight
                di = [
                    (dij, _weight(j), y0j)
                    for j, (dij, y0j)
                    in enumerate(sorted(di, key=lambda _: _[0]), start=1)
                ]
            elif weighted == 2:
                if weight is None:
                    di.sort(key=lambda _: _[0])
                    di1, dik = di[0][0], di[k - 1][0]
                    if dik != di1:
                        def _weight(d):
                            return (dik - d) / (dik - di1)
                    else:
                        def _weight(d):
                            return 1
                else:
                    _weight = weight
                di = [(dij, _weight(dij), y0j) for dij, y0j in di]
            for _, wij, yi in sorted(di, key=lambda _: _[0])[:k]:
                ci[yi] += wij
        else:
            for _, yi in sorted(di, key=lambda _: _[0])[:k]:
                ci[yi] += 1

        a.append(sorted(ci, key=ci.get)[-1])
    return a


class KNN:
    def fit(
        self,
        X0: list[list],
        y0: list,
        weighted: int = 0,
        distance: object = euclidean_distance,
        weight: object = None
    ):
        self.X0 = X0
        self.y0 = y0
        self.weighted = weighted
        self.distance = distance
        self.weight = weight

    def predict(self, X: list[list], k: int) -> list:
        return predict(
            X, self.X0, self.y0,
            k, self.weighted,
            self.distance,  self.weight
        )
