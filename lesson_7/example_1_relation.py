from typing import Union


# Distance metrics.

def euclidean_distance(x: list[float], y: list[float]) -> float:
    return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)]) ** .5


# Classification quality metrics.

def accuracy(a: list, y: list) -> Union[float, None]:
    n = sum([ai == yi for ai, yi in zip(a, y)])
    d = len(a)
    return n / d if d else None


# def precision(a: list, y: list) -> Union[float, None]:
#     n = sum([ai == yi for ai, yi in zip(a, y) if ai])
#     d = len([ai for ai in a if ai])
#     return n / d if d else None


# def recall(a: list, y: list) -> Union[float, None]:
#     n = sum([ai == yi for ai, yi in zip(a, y) if ai])
#     d = len([yi for yi in y if yi])
#     return n / d if d else None


# def fscore(a: list, y: list) -> Union[float, None]:
#     p = precision(a, y)
#     r = recall(a, y)
#     return 2 * p * r / (p + r) \
#         if p is not None and r is not None and p + r else None


# Error matrix.

# def error_matrix(a: list, y: list) -> tuple[tuple[int, int], tuple[int, int]]:
#     TP = FP = TN = FN = 0
#     for ai, yi in zip(a, y):
#         if ai:
#             if ai == yi:
#                 TP += 1
#             else:
#                 FP += 1
#         else:
#             if ai == yi:
#                 TN += 1
#             else:
#                 FN += 1
#     return ((TP, FP), (FN, TN))


# def error_accuracy(
#     e: tuple[tuple[int, int], tuple[int, int]]
# ) -> Union[float, None]:
#     ((TP, FP), (FN, TN)) = e
#     return (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else None


# def error_precision(
#     e: tuple[tuple[int, int], tuple[int, int]]
# ) -> Union[float, None]:
#     ((TP, FP), _) = e
#     return TP / (TP + FP) if TP + FP else None


# def error_recall(
#     e: tuple[tuple[int, int], tuple[int, int]]
# ) -> Union[float, None]:
#     ((TP, FN), _) = zip(*e)
#     return TP / (TP + FN) if TP + FN else None


# def error_fscore(
#     e: tuple[tuple[int, int], tuple[int, int]]
# ) -> Union[float, None]:
#     ((TP, FP), (FN, _)) = e
#     p = TP / (TP + FP) if TP + FP else None
#     r = TP / (TP + FN) if TP + FN else None
#     return 2 * p * r / (p + r) \
#         if p is not None and r is not None and p + r else None
