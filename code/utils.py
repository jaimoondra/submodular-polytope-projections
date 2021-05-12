import math
import numpy as np
import random
from typing import List, Dict, Set
from constants import *


def round_list(x: List[float], accuracy: int = base_decimal_accuracy):
    """
    Round a list of numbers to appropriate accuracy
    :param x: List of numbers
    :param accuracy: Number of decimal digits to round float numbers to
    :return: List of rounded numbers
    """
    x = [round(i, accuracy) for i in x]
    return x


def sort(x: List[float], reverse=False):
    """
    :param x: List of numbers
    :param reverse: Sorts in decreasing order if set to True
    :return: Sorted list and the corresponding mapping (permutation)
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
    return {mapping[i]: i for i in range(len(mapping))}


def map_set(S: Set[int], mapping: Dict[int, int]):
    """
    Determines the range of S under mapping
    :param S: set of integers
    :param mapping: mapping
    :return: range of S under mapping as a set
    """
    return set({mapping[i] for i in S})


def permute(x: List[float], mapping: Dict[int, int]):
    """
    Permutes x according to mapping
    :param x:
    :param mapping:
    :return:
    """
    y = [0.0] * len(x)
    for i in range(len(x)):
        y[mapping[i]] = x[i]

    return y


def generate_random_permutation(n: int):
    A = list(range(n + 1)[1: n + 1])
    random.shuffle(A)
    return A