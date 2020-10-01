import numpy as np
from typing import List, Dict, Set


def sort(x: List[int], reverse=False):
    """
    :param x: List of numbers
    :param reverse: Sorts in decreasing order if set to True
    :return: Sorted list and the corresponding sorted indices
    """
    enum = sorted(enumerate(x), key=lambda z: z[1], reverse=reverse)
    y = [enum[j][1] for j in range(len(enum))]
    mapping = {enum[j][0]: j for j in range(len(enum))}

    return y, mapping


def invert(mapping: Dict[int, int]):
    """
    Invert a (bijective) mapping {0, ..., n - 1} -> {0, ..., n - 1}
    :param mapping: Original mapping
    :return: Inverse of the original mapping
    """
    n = len(mapping)
    return {mapping[i]: i for i in range(n)}


def map_set(S: Set[int], mapping: Dict[int, int]):
    return set({mapping[i] for i in S})


def permute(x: List[int], mapping: Dict[int, int]):
    y = [0] * len(x)
    print(mapping)
    for i in range(len(x)):
        y[mapping[i]] = x[i]

    return y


x = [10, 10000, 100, 1000]
y, m = sort(x, reverse=True)
print(y, m)
m1 = invert(m)
z = permute(y, m1)
print(z)
