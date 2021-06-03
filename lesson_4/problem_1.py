from example_1 import Node, Leaf, find_best_split, split


def build_tree(
    data, labels,
    level: int = 0, level_max: int = -1
):
    if level_max >= 0 and level == level_max:
        return Leaf(data, labels)

    quality, t, index = find_best_split(data, labels)

    if quality == 0:
        return Leaf(data, labels)

    true_data, false_data, true_labels, false_labels = \
        split(data, labels, index, t)

    true_branch = build_tree(
        true_data, true_labels,
        level=level + 1, level_max=level_max
    )
    false_branch = build_tree(
        false_data, false_labels,
        level=level + 1, level_max=level_max
    )

    return Node(index, t, true_branch, false_branch)


def error_matrix(z, y):
    e = [[0, 0], [0, 0]]
    for zi, yi in zip(z, y):
        if zi:
            if zi == yi:
                e[0][0] += 1
            else:
                e[0][1] += 1
        else:
            if zi == yi:
                e[1][1] += 1
            else:
                e[1][0] += 1
    return e


def error_metrics(e):
    [[TP, FP], [FN, TN]] = e
    a = (TP + TN) / (TP + TN + FP + FN)
    p = TP / (TP + FP) if TP + FP else None
    r = TP / (TP + FN) if TP + FN else None
    f = 2 * p * r / (p + r) if p is not None and r is not None else None
    return a, p, r, f
