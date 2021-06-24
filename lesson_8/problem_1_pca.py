import numpy as np


def pca_eigens(X: np.array) -> np.array:
    return sorted(
        zip(
            *(lambda _: (_[0], _[1].transpose()))(
                np.linalg.eig(np.dot(X.transpose(), X))
            )
        ),
        key=lambda _: abs(_[0]),
        reverse=True
    )


def pca_singulars(X: np.array) -> np.array:
    return zip(*np.linalg.svd(X)[1:])


def pca(X: np.array, d: int) -> np.array:
    return np.dot(
        X,
        np.linalg.svd(X)[-1][:d, :].transpose()
    )


# def pca(X: np.array, d: int) -> np.array:
#     return np.dot(
#         X,
#         np.array(
#             [
#                 _[1] for _ in sorted(
#                     zip(
#                         *(lambda _: (_[0], _[1].transpose()))(
#                             np.linalg.eig(np.dot(X.transpose(), X))
#                         )
#                     ),
#                     key=lambda _: abs(_[0]),
#                     reverse=True
#                 )[:d]
#             ]
#         ).transpose()
#     )
