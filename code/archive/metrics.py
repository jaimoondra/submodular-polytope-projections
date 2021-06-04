from typing import List
from inc_fix import CardinalityPolytope
import numpy as np
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def l_norm_violated_constraints(l: float, card_pol: CardinalityPolytope, y: List[float]):
    """
    l-norm of violated constraints: If Ax <= b is the set of constraints for the polytope, then ||{Ay - b}_+||_l
    :param l: nonnegative real number or infinity. The 'l' in the l-norm
    :param card_pol: Cardinality Polytope
    :param y: Point in space
    """
    y.sort(reverse=True)

    violated_constraints_values = []
    for S in powerset(range(len(card_pol))):
        y_S = 0
        count = 0
        for i in S:
            y_S += y[i]
            count += 1

        if y_S > card_pol.g[len(S)]:
            violated_constraints_values.append(y_S - card_pol.g[len(S)])

    return np.linalg.norm(violated_constraints_values, l)


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
