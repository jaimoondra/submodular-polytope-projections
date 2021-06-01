import logging
logging.basicConfig(level=logging.WARNING)

from continuous_methods import *
from utils import *
from constants import *


n = 50

f = PermutahedronSubmodularFunction(n)
P = CardinalityPolytope(f)

np.random.seed(1)
random.seed(1)

total_time_unoptimized = 0.0
total_time_optimized = 0.0
total_iterations_unoptimized = 0
total_iterations_optimized = 0

for j in range(20):
    logging.warning('Point ' + str(j))
    y= 2 * n * np.random.random(n)
    h = lambda x: 0.5 * np.dot(x - y, x - y)
    grad_h = lambda x: np.power(x - y, 1)
    h_tol, time_tol = -1, np.inf
    epsilon = 3/(n * n * n)
    lmo = lambda c: P.linear_optimization_over_base(c)[1]

    w = generate_random_permutation(n)
    S = {tuple(w): 1}

    g = P.f.g
    g = {i: g[i] for i in range(n + 1)}
    x_star = isotonic_projection(y, g)
    actual_tight_sets = {s[1] for s in determine_tight_sets(y, x_star)}.union({frozenset()})


    logging.warning('Starting unoptimized AFW.')
    x1, function_value1, time1, t1, primal_gap1, S1 =\
        AFW(np.array(w), S, lmo, epsilon, h, grad_h, h_tol, time_tol)

    logging.warning('Starting cut optimized AFW.')
    x2, function_value2, time2, t2, primal_gap2, S2, inferred_tight_sets =\
        adaptive_AFW_cardinality_polytope(np.array(w), S, P, epsilon, h, grad_h, h_tol, time_tol,
                                          set(), y)

    # print(x_star, x1, x2)
    logging.warning(str(np.linalg.norm(x_star - x1)) + str(' ') +
                    str(np.linalg.norm(x_star - x2)) + str(' ') +
                    str(np.linalg.norm(x1 - x2)))

    logging.warning(str(sum(time1)) + ' ' + str(sum(time2)))
    total_time_unoptimized = total_time_unoptimized + sum(time1)
    total_time_optimized = total_time_optimized + sum(time2)
    total_iterations_unoptimized = total_iterations_unoptimized + t1
    total_iterations_optimized = total_iterations_optimized + t2
    logging.warning(str(t1) + ' ' + str(t2))
    #print(len(inferred_tight_sets), inferred_tight_sets)
    #print(len(actual_tight_sets), actual_tight_sets)
    #print(actual_tight_sets.difference(inferred_tight_sets))


logging.warning(str(total_time_unoptimized) + ' ' + str(total_time_optimized))
logging.warning(str(total_iterations_unoptimized) + ' ' + str(total_iterations_optimized))
