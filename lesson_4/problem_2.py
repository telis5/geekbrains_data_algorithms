from __future__ import annotations
from typing import Union, Any
from math import log2


class Node:
    def __init__(
            self, j: int, x,
            branch_true: Union[Node, Leaf], branch_false: Union[Node, Leaf]
    ):
        self.j = j
        self.x = x
        self.branch_true = branch_true
        self.branch_false = branch_false


class Leaf:
    def __init__(self, X: list[list], y: list, func_predict: object):
        self.X = X
        self.y = y
        self._predict = func_predict
        self.a = self.predict()

    def predict(self):
        return self._predict(self.y)


def assort(x: list[Any]) -> dict[Any: int]:
    a = {}
    for xi in x:
        if xi not in a:
            a[xi] = 1
        else:
            a[xi] += 1
    return a


def predict_argmax(x: list[Any]) -> Any:
    a = assort(x)
    return max(a, key=a.get)


def predict_probabilities(x: list[Any]) -> dict[Any: float]:
    a = assort(x)
    return {xi: ni / len(x) for xi, ni in a.items()}


def predict_mean(x: list[Union[int, float]]) -> float:
    return sum(x) / len(x)


def impurity_gini(p: list[float]) -> float:
    return 1 - sum([pi ** 2 for pi in p])


def information_gain(p: list[float]) -> float:
    return - sum([pi * log2(pi) for pi in p])


def impurity(x: list, func: object) -> float:
    return func([k / len(x) for k in assort(x).values()])


def quality(y1: list, y0: list, H0: float, func: object) -> float:
    def _impurity(x: list) -> object:
        return impurity(x, func=func)

    p1 = len(y1) / (len(y1) + len(y0))
    p0 = 1 - p1
    return H0 - p1 * _impurity(y1) - p0 * _impurity(y0)


def condition_threshold(x: Union[int, float], t: Union[int, float]) -> bool:
    return x <= t


def segregate(x: list, cond: object) -> tuple[list[int], list[int]]:
    s1, s0 = [], []
    for i in range(len(x)):
        if cond(x[i]):
            s1.append(i)
        else:
            s0.append(i)
    return s1, s0


def split(x: list[Any], S: list[list[int]]) -> list[list[Any]]:
    return [[x[i] for i in s] for s in S]


def column(X: list[list], j: int) -> list:
    return [xi[j] for xi in X]


def fit(
    X: list[list[Any]], y: list,
    func_impurity: object, cond_segregate: object,
    len_min: int = 5
) -> tuple[float, Any, int]:
    def _quality(y1: list, y0: list) -> object:
        return quality(
            y1, y0, H0=impurity(y, func=func_impurity), func=func_impurity
        )

    Qmax, j_Qmax, x_Qmax = 0, None, None
    for j in range(len(X[0])):
        xj = column(X, j)
        for xjk in set(xj):
            s1, s0 = segregate(xj, cond=(lambda _: cond_segregate(_, xjk)))

            if len(s1) < len_min or len(s0) < len_min:
                continue

            y1, y0 = split(y, (s1, s0))
            Qjk = _quality(y1, y0)

            if Qjk > Qmax:
                Qmax, j_Qmax, x_Qmax = Qjk, j, xjk

    return Qmax, j_Qmax, x_Qmax


def build(
    X: list[list], y: list,
    lev: int = 0,
    func_impurity: object = impurity_gini,
    cond_segregate: object = condition_threshold,
    func_predict: object = predict_argmax,
    lev_max: int = -1,
    len_min: int = 5
) -> Union[Node, Leaf]:
    def _fit(X: list[list[Any]], y: list) -> object:
        return fit(
            X, y,
            func_impurity=func_impurity,
            cond_segregate=cond_segregate,
            len_min=len_min
        )

    def _build(X: list[list], y: list, lev: int) -> object:
        return build(
            X, y, lev,
            func_impurity=func_impurity,
            cond_segregate=cond_segregate,
            func_predict=func_predict,
            lev_max=lev_max,
            len_min=len_min
        )

    if lev_max >= 0 and lev == lev_max:
        return Leaf(X, y, func_predict=func_predict)

    Qmax, j_Qmax, x_Qmax = _fit(X, y)

    if Qmax == 0:
        return Leaf(X, y, func_predict=func_predict)

    xj = column(X, j_Qmax)
    s1, s0 = segregate(xj, cond=(lambda _: cond_segregate(_, x_Qmax)))
    X1, X0 = split(X, (s1, s0))
    y1, y0 = split(y, (s1, s0))

    b1 = _build(X=X1, y=y1, lev=lev + 1)
    b0 = _build(X=X0, y=y0, lev=lev + 1)

    return Node(j=j_Qmax, x=x_Qmax, branch_true=b1, branch_false=b0)


def classify(xi: list, node: Union[Node, Leaf], cond: object):
    if isinstance(node, Leaf):
        return node.a

    if cond(xi[node.j], node.x):
        return classify(xi, node.branch_true, cond=cond)
    else:
        return classify(xi, node.branch_false, cond=cond)


def predict(
    X: list[list], tree: Union[Node, Leaf], cond: object = condition_threshold
) -> list:
    return [classify(xi, tree, cond=cond) for xi in X]


def print_tree(
    node: Union[Node, Leaf], spac: str = '', spac_lev: str = '\t'
):
    if isinstance(node, Leaf):
        print(spac + 'prediction:', node.a)
        return

    print(spac + 'index:', str(node.j))
    print(spac + 'value:', str(node.x))

    print(spac + 'true:')
    print_tree(node=node.branch_true, spac=spac + spac_lev)

    print(spac + 'false:')
    print_tree(node=node.branch_false, spac=spac + spac_lev)


def error_matrix(a: list, y: list) -> tuple[tuple[int, int], tuple[int, int]]:
    TP = FP = TN = FN = 0
    for ai, yi in zip(a, y):
        if ai:
            if ai == yi:
                TP += 1
            else:
                FP += 1
        else:
            if ai == yi:
                TN += 1
            else:
                FN += 1
    return ((TP, FP), (FN, TN))


def error_metrics(
    e: tuple[tuple[int, int], tuple[int, int]]
) -> tuple[float, float, float, float]:
    ((TP, FP), (FN, TN)) = e
    a = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else None
    p = TP / (TP + FP) if TP + FP else None
    r = TP / (TP + FN) if TP + FN else None
    f = 2 * p * r / (p + r) \
        if p is not None and r is not None and p + r else None
    return a, p, r, f
