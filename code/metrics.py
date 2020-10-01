from typing import List
from inc_fix import CardinalityPolytope
import numpy as np


def l_norm_violated_constraints(l: float, card_pol: CardinalityPolytope, y: List[float]):
    """
    l-norm of violated constraints: If Ax <= b is the set of constraints for the polytope, then ||{Ay - b}_+||_l
    :param l: nonnegative real number or infinity. The 'l' in the l-norm
    :param card_pol: Cardinality Polytope
    :param y: Point in space
    """
    return


def binary_search(v: List[float], t: float):
    """
    Returns index i such that v_0, ..., v_i <= t
    :param v: list of real numbers, sorted in an increasing sequence
    :param t: real number
    :return: index i such that v_0, ..., v_{i - 1} <= t, v_j > t for j >= i
    """
    l, u = 0, len(v)

    while l < u:
        b = int((l + u) / 2)
        if v[b] > t:
            u = b
        else:
            l = b + 1

    return l