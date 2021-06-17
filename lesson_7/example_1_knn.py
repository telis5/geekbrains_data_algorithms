class KNN:
    @staticmethod
    def _predict(
        X: list[list],
        X0: list[list],
        y0: list,
        metric: object,
        k: int
    ) -> list:
        a = []
        for xi in X:
            di = [(metric(xi, x0j), y0j) for x0j, y0j in zip(X0, y0)]
            ci = {_: 0 for _ in set(y0)}
            for _, yi in sorted(di)[:k]:  # sorted(di, key=lambda _: _[0])[:k]
                ci[yi] += 1
            a.append(sorted(ci, key=ci.get)[-1])
        return a

    def fit(self, X0: list[list], y0: list, metric: object):
        self.X0 = X0
        self.y0 = y0
        self.metric = metric

    def predict(self, X: list[list], k: int) -> list:
        return self._predict(X, self.X0, self.y0, self.metric, k)
