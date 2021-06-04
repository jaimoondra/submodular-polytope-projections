from typing import List, Dict, Set
import numpy as np
import math
import logging
from inc_fix import CardinalityPolytope, IncFix, permute, sort, invert, map_set
from constants import *
from close_points import generate_concave_function, random_point

logging.basicConfig(level=logging.WARNING)


def determine_observed_tight_sets(y, points, gradient_differences, observed_tight_sets,
                                  tight_sets, gradients):
    """
    :param y:
    :param points:
    :param gradient_differences:
    :param observed_tight_sets:
    :param tight_sets:
    :return:
    """
    n = len(y)
    flag_tight_set_seen = {frozenset(tight_set): 0 for tight_set in tight_sets}
    for j in range(len(gradient_differences) - 1):
        # d = round(max([abs(y[i] - points[j][i]) for i in range(n)]),
        #          base_decimal_accuracy)
        d = np.linalg.norm(np.array(y) - np.array(points[j]))

        for k in range(len(gradient_differences[j])):
            if gradient_differences[j][k][0] > 4 * d:
                if frozenset(gradient_differences[j][k][1]) not in flag_tight_set_seen.keys():
                    print(y[27], points[j][27], y[27] - points[j][27], n)
                    print('Distance: ', d)
                    print('Gradient difference: ', gradient_differences[j][k][0])
                    print('Tight set in question: ', gradient_differences[j][k][1])
                    print('First point y')
                    print('Gradient: ', {i: gradients[j][i] for i in range(n)})
                    print('y: ', {i: points[j][i] for i in range(n)})
                    print('x: ', {i: round(gradients[j][i] + points[j][i], base_decimal_accuracy)
                                  for i in range(n)})
                    print('Second point y tilde')
                    print('Gradient: ', {i: gradients[-1][i] for i in range(n)})
                    print('y: ', {i: y[i] for i in range(n)})
                    print('x: ', {i: round(gradients[-1][i] + y[i], base_decimal_accuracy)
                                  for i in range(n)})
                    print('Tight sets: ', tight_sets)
                    raise ValueError('COMPARE POINTS Tight set not seen previously.')
                flag_tight_set_seen[frozenset(gradient_differences[j][k][1])] = 1

    seen = 0
    for tight_set in tight_sets:
        if flag_tight_set_seen.get(frozenset(tight_set)) == 1:
            seen = seen + 1

    observed_tight_sets[0].append(seen)
    observed_tight_sets[1].append(len(tight_sets))

    return


def map_gradient_differences_to_tight_sets(gradient: List[float], tight_sets: List[Set[int]]):
    unique_gradients = sorted(set(gradient))

    return [[round(unique_gradients[j + 1] - unique_gradients[j], base_decimal_accuracy),
            tight_sets[j + 1]] for j in range(len(tight_sets) - 2)]


def close_points_learning(n: int, m: int = 40, seed=None):
    """
    Uses algorithm described after Lemma 7 to determine save in running time
    :param n: Size of the ground set
    :param m: Number of points in the neighborhood
    :param seed: seed for random
    """
    np.random.seed(seed)

    # Submodular function f(S) := g(|S|)
    g = generate_concave_function(n)

    # Table for storing points:
    points, gradient_differences = [], []

    # Central random point
    print('Point #' + str(0))
    y0 = random_point(n, std_dev_point, True, seed=None)
    points.append(y0)

    # Calculate projection
    proj = IncFix(g, y0)
    proj.projection()
    tight_sets0, gradient0 = proj.tight_sets, proj.gradient
    print(tight_sets0, gradient0)

    # Table for storing gradient differences and tight sets
    gradients = [gradient0]

    # unique_gradient0 = sorted(set(blunt_sharp_edges(sorted(gradient0))))
    if len(set(gradient0)) != len(tight_sets0) - 1:
        print('Danger!', len(gradient0), len(tight_sets0))

    gradient_differences.append(map_gradient_differences_to_tight_sets(gradient0, tight_sets0))

    observed_tight_sets = [[], []]
    for i in range(1, m + 1):
        print('Point #' + str(i))

        error = random_point(n, 1/math.pow(n, 1), True, seed=None)
        y = [round(abs(y0[i] + error[i]), base_decimal_accuracy) for i in range(n)]
        points.append(y)

        proj = IncFix(g, y)
        proj.projection()
        tight_sets, gradient = proj.tight_sets, proj.gradient
        gradients.append(gradient)

        if len(set(gradient)) != len(tight_sets) - 1:
            print('Danger!', len(set(gradient)), len(tight_sets))
            sorted_gradient, mapping = sort(gradient)
            print({i: sorted_gradient[i] for i in range(n)})
            sorted_tight_sets = [map_set(S, mapping) for S in tight_sets]
            print({i: sorted_gradient[i] for i in range(n)})
            print(sorted_tight_sets)
            print([sorted_tight_sets[i + 1] - sorted_tight_sets[i] for i in range(
                len(sorted_tight_sets) - 1)])

        gradient_differences.append(map_gradient_differences_to_tight_sets(gradient, tight_sets))
        determine_observed_tight_sets(y, points, gradient_differences, observed_tight_sets,
                                      tight_sets, gradients)

    return observed_tight_sets
