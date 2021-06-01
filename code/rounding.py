from gurobipy import *
import numpy as np
from submodular_polytope import CardinalitySubmodularFunction, CardinalityPolytope, \
    PermutahedronSubmodularFunction, Permutahedron
from typing import List, Set
from constants import *
from utils import *


def active_set_oracle(vertices, x):
    try:
        m = Model("opt")
        n = len(vertices[0])

        # define variables
        lam = {}
        for i in range(len(vertices)):
            lam[i] = m.addVar(lb=0, name='lam_{}'.format(i),obj = 1)

        # feasibility constraints
        for i in range(n):
            m.addConstr(x[i],'=', sum([lam[j] * vertices[j][i] for j in range(len(vertices))]))

        # convex hull constraint
        m.addConstr(quicksum([lam[i] for i in lam.keys()]), '=',1)
        m.update()

        # optimize
        m.setParam( 'OutputFlag', False )
        # m.write('exact.lp')
        m.optimize()
        v = {}

        for i in lam:
            if np.round(lam[i].x,5) > 0:
                v[tuple(vertices[i])] = lam[i].x

        return True

    except AttributeError:
        return False


def tight_set_rounded_solution(C: CardinalityPolytope, H: List[Set], y):
    """
    T is assumed to contain the empty set and the ground set, and is assumed to be a chain.
    :param C:
    :param T:
    :return:
    """
    n = len(C)
    x = np.zeros(n)

    for j in range(1, len(H)):
        F = H[j].difference(H[j - 1])
        y_F = sum([y[k] for k in F])
        alpha = (C.f.function_value(H[j]) - C.f.function_value(H[j - 1]) - y_F)/len(F)
        for i in F:
            x[i] = alpha + y[i]

    return x


def cut_rounding(C: CardinalityPolytope, H: List[Set], y, S: Dict):
    H.sort(key=len)

    x = tight_set_rounded_solution(C, H, y)
    is_convex_combination = active_set_oracle(list(S.keys()), x)

    if is_convex_combination:
        return x
    else:
        raise ValueError('Cannot round, x not feasible.')


# n = 10
# f = PermutahedronSubmodularFunction(n)
# P = CardinalityPolytope(f)
# H = [set(), {0, 1, 2}, {0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}]
# y = [0, 0, 1, 0, 0, 0, 1, 0, 0, 3]
# print(tight_set_rounded_solution(P, H, y))
