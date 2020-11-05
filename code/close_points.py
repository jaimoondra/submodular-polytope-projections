from typing import List, Dict, Set
import numpy as np
import math
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from inc_fix import CardinalityPolytope, IncFix, permute, sort, invert, map_set
from constants import *
from metrics import l_norm_violated_constraints


logging.basicConfig(level=logging.INFO)


def swati_thesis_example():
    """
    Example from Swati's thesis
    """
    g = [0.5, 0.6, 0.7, 0.8, 0.87]
    y = [1, 0.9, 0.9, 0.9, 1.1]
    proj = IncFix(g, y)
    x_star, iterates = proj.projection()
    print(x_star, iterates)


def generate_concave_function(n: int, seed=None):
    """
    Return a 'random' concave function
    :param n: Dimension of the ground set
    :param seed: Seed for random number generators
    :return: A concave function (a.k.a. cardinality function)
    """
    if seed is not None:
        np.random.seed(seed)

    g = random_point(n, 1, True, seed)
    g = [round(x, base_decimal_accuracy) for x in g]
    g.sort(reverse=True)

    for i in range(1, len(g)):
        g[i] = g[i - 1] + g[i]

    g = round_list(g, base_decimal_accuracy)

    return g


def round_list(x: List[float], accuracy: int = base_decimal_accuracy):
    """
    Round a list of numbers to appropriate accuracy
    :param x: List of numbers
    :param accuracy: Number of decimal digits to round float numbers to
    :return: List of rounded numbers
    """
    x = [round(i, accuracy) for i in x]
    return x


def plot_iterates(iterates: List[List[float]], y: List[float], color1, color2):
    """
    Plot for iterates of the inc-fix algorithm.
    x-axis: elements of the ground set, in increasing order (0, 1, ..., N - 1)
    y-axis: value of the iterate
    That is, point (n, v_n) represents the value of point v = (v_0, ..., v_{N - 1}) at coordinate n.
    Several iterates are plotted in the same figure, with the original point y in a different color.
    :param iterates: List of points (iterates)
    :param y: The original point y
    :param color1, color2: first color. One of 'r', 'g', 'c' etc. See https://matplotlib.org/2.0.2/api/colors_api.html
    for the full list.
    """
    jump = max(int(n / 10), min(5, int(n / 2)))
    for i in range(len(iterates)):
        iterates[i] = round_list(iterates[i], base_decimal_accuracy)
        if i % jump == 1 or i == len(iterates) - 1:
            plt.plot(iterates[i], color1)

    plt.xlabel('Ground set elements')
    plt.ylabel('Value')
    plt.plot(y, color2)


def plot_tight_sets(tight_sets: List[List[int]], color1):
    plt.xlabel('Iteration')
    plt.ylabel('Tight set size')
    plt.plot(tight_sets, color1)


def random_point(n: int, r: float = 1, nonnegative: bool = False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    z = np.random.multivariate_normal(mean=[0] * n, cov=r * np.identity(n))
    if nonnegative:
        z = [round(x, base_decimal_accuracy) if x >= 0 else round(-x, base_decimal_accuracy) for x in z]

    return z


def map_items(iterates: List[List[float]], y: List[float], x_star: List[float], tight_sets: List[Set[int]],
              mapping: Dict[int, int]):
    y1 = permute(y, mapping)
    x_star = permute(x_star, mapping)
    for i in range(len(iterates)):
        iterates[i] = permute(iterates[i], mapping)
    for i in range(len(tight_sets)):
        tight_sets[i] = map_set(tight_sets[i], mapping)

    return iterates, y1, x_star, tight_sets


def close_points(n: int, reps: int = 40):
    np.random.seed(seeds[0])
    g = generate_concave_function(n)

    # Central random point
    y = random_point(n, std_dev_point, True, seed=None)

    proj = IncFix(g, y)
    mapping = proj.mapping
    x_star, iterates, tight_sets = proj.projection()

    tight_sets_for_multiple_points = []
    iterates, y1, x_star, tight_sets = map_items(iterates, y, x_star, tight_sets, mapping)

    T = 0
    for i in range(reps):
        print('Rep ' + str(i))
        noise = random_point(n, (1/(n*n)), False, seed=None)
        y0 = [round(abs(y[i] + noise[i]), base_decimal_accuracy) for i in range(n)]

        proj0 = IncFix(g, y0)
        x_star0, iterates0, tight_sets0 = proj0.projection()
        iterates0, y0, x_star0, tight_sets0 = map_items(iterates0, y0, x_star0, tight_sets0, mapping)
        tight_sets_for_multiple_points.append(tight_sets0)

    tight_set_accuracy = []
    t, T = 0, 0
    for iteration in range(len(tight_sets)):
        tight_set1 = tight_sets[iteration]
        for rep in range(reps):
            if len(tight_sets_for_multiple_points[rep]) > iteration:
                tight_set2 = tight_sets_for_multiple_points[rep][iteration]
                if len(tight_set1) > 0 or len(tight_set2) > 0:
                    t += 2 * len(tight_set2.intersection(tight_set1))
                    T += len(tight_set1) + len(tight_set2)

        if T > 0:
            tight_set_accuracy.append(round(t/T, base_decimal_accuracy))

    # plt.plot(tight_set_accuracy)
    # plt.xlabel('iteration')
    # plt.ylabel('Tight sets match')
    # plt.show()
    # plt.clf()

    return tight_set_accuracy


def random_points(n: int, reps: int = 40):
    np.random.seed(seeds[0])
    g = generate_concave_function(n)

    # Central random point
    y = random_point(n, std_dev_point, True, seed=None)

    proj = IncFix(g, y)
    mapping = proj.mapping
    x_star, iterates, tight_sets = proj.projection()

    tight_sets_for_multiple_points = []
    iterates, y, x_star, tight_sets = map_items(iterates, y, x_star, tight_sets, invert(mapping))

    for i in range(reps):
        print('Rep ' + str(i))
        y0 = random_point(n, std_dev_point, True, seed=None)

        proj0 = IncFix(g, y0)
        x_star0, iterates0, tight_sets0 = proj0.projection()
        iterates0, y0, x_star0, tight_sets0 = map_items(iterates0, y0, x_star0, tight_sets0, invert(mapping))
        tight_sets_for_multiple_points.append(tight_sets0)

    tight_set_accuracy = []
    t, T = 0, 0
    for iteration in range(len(tight_sets)):
        tight_set1 = tight_sets[iteration]
        for rep in range(reps):
            if len(tight_sets_for_multiple_points[rep]) > iteration:
                tight_set2 = tight_sets_for_multiple_points[rep][iteration]
                if len(tight_set1) > 0 or len(tight_set2) > 0:
                    t += 2 * len(tight_set2.intersection(tight_set1))
                    T += len(tight_set1) + len(tight_set2)

        if T > 0:
            tight_set_accuracy.append(round(t/T, base_decimal_accuracy))

    # plt.plot(tight_set_accuracy)
    # plt.xlabel('iteration')
    # plt.ylabel('Tight sets match')
    # plt.show()
    # plt.clf()

    return tight_set_accuracy


def compare_points(n: int):
    # Cardinality function a.k.a positive concave function
    seed = seeds[0]
    g = generate_concave_function(seed)
    print('g = ' + str(g))

    # First random point
    seed = seeds[1]
    np.random.seed(seed)
    y0 = random_point(n, 1.5, True, seed)
    y0 = [round(i, base_decimal_accuracy) for i in y0]
    print('y0 = ' + str(y0))

    proj = IncFix(g, y0)
    x_star0, iterates0, tight_sets0 = proj.projection()
    x_star0 = round_list(x_star0, base_decimal_accuracy)
    print('projection of y0 on P(f) = ' + str(x_star0))

    # # Point close to first random point
    # noise = random_point(n, (1 / (n * n)), False, seed)
    # y1 = [round(y0[i] + noise[i], base_decimal_accuracy) for i in range(n)]
    # y1.sort(reverse=True)
    # print('y1 = ' + str(y1))
    #
    # proj = IncFix(g, y1)
    # x_star1, iterates1, tight_sets1 = proj.projection()
    # x_star1 = round_list(x_star1, base_decimal_accuracy)
    # print('projection of y1 on P(f) = ' + str(x_star1))

    # # First random point
    # seed = seeds[2]
    # np.random.seed(seed)
    # y2 = random_point(n, 3, True, seed=seed)
    # y2 = [round(i, base_decimal_accuracy) for i in y2]
    # print('y2 = ' + str(y2))
    #
    # proj = IncFix(g, y2)
    # x_star2, iterates2, tight_sets2 = proj.projection()
    # x_star2 = round_list(x_star2, base_decimal_accuracy)
    # print('projection of y2 on P(f) = ' + str(x_star2))

    # Plots - close points
    plt.title('Iterates with time - 2 close points')
    plot_iterates(iterates0, y0, 'b', 'c')
    # plot_iterates(iterates1, y1, 'r', 'm')
    # plt.savefig('figures\\' + str(n) + '-close-points-iterates.png')
    plt.show()
    plt.clf()

    # plt.title('Size of tight sets with time - 2 close points')
    # plot_tight_sets(tight_sets0, 'b')
    # plot_tight_sets(tight_sets1, 'r')
    # plt.savefig('figures\\' + str(n) + '-close-points-tight-sets.png')
    # plt.clf()
    #
    # # Plots - two random points
    # plt.title('Iterates with time - 2 random points')
    # plot_iterates(iterates0, y0, 'b', 'c')
    # plot_iterates(iterates2, y2, 'g', 'y')
    # plt.savefig('figures\\' + str(n) + '-random-points-iterates.png')
    # plt.clf()
    #
    # plt.title('Size of tight sets with time - 2 random points')
    # plot_tight_sets(tight_sets0, 'b')
    # plot_tight_sets(tight_sets2, 'g')
    # plt.savefig('figures\\' + str(n) + '-random-points-tight-sets.png')
    # plt.clf()


def compare_points_unsorted(n: int):
    # Cardinality function a.k.a positive concave function
    seed = seeds[0]
    g = generate_concave_function(seed)
    print('g = ' + str(g))

    # First random point
    seed = seeds[1]
    np.random.seed(seed)
    y = random_point(n, 3, True, seed)
    y = [round(i, base_decimal_accuracy) for i in y]
    print('y0 = ' + str(y))
    y0, indices0 = decreasing_sort(y)

    proj = IncFix(g, y0)
    x_star0, iterates0, tight_sets0 = proj.projection()
    x_star0 = restore(x_star0, indices0)

    x_star0 = round_list(x_star0, base_decimal_accuracy)
    print('projection of y0 on P(f) = ' + str(x_star0))

    # Point close to first random point
    noise = random_point(n, (1 / (n * n)), False, seed)
    y1 = [round(y[i] + noise[i], base_decimal_accuracy) for i in range(n)]
    print('y1 = ' + str(y1))
    y1, indices1 = decreasing_sort(y1)
    proj = IncFix(g, y1)
    x_star1, iterates1, tight_sets1 = proj.projection()

    # Align y_1, x_1* and iterates_1 with y_0, x_0* and iterates_0 resp.
    y1 = restore(y1, indices1)
    y1 = inverse_restore(y1, indices0)

    x_star1 = restore(x_star1, indices1)
    x_star1 = inverse_restore(x_star1, indices0)

    for i in range(len(iterates1)):
        iterates1[i] = restore(iterates1[i], indices1)
        iterates1[i] = inverse_restore(iterates1[i], indices0)

    x_star1 = round_list(x_star1, base_decimal_accuracy)
    print('indices = ' + str(indices1))
    print('projection of y1 on P(f) = ' + str(x_star1))

    # Plots - close points
    plt.title('Iterates with time - 2 close points')
    plot_iterates(iterates0, y0, 'b', 'c')
    plot_iterates(iterates1, y1, 'r', 'm')
    # plt.show()
    plt.savefig('figures\\unsorted\\' + str(n) + '-close-points-iterates.png')
    plt.clf()


def close_points_unsorted(n: int, reps: int = 4):
    np.random.seed()
    g = generate_concave_function()
    print('g = ' + str(g))

    # Central random point
    y, iterates, tight_sets, x_stars = [], [], [], []
    y0 = random_point(n, 1.5, True, seed=None)
    y0 = [round(i, base_decimal_accuracy) for i in y0]
    y0, indices = decreasing_sort(y0)
    print('y0 = ' + str(y0))
    y.append(y0)

    proj = IncFix(g, y0)
    x_star0, iterates0, tight_sets0 = proj.projection()
    x_star0 = restore(x_star0, indices)
    x_star0 = round_list(x_star0, base_decimal_accuracy)
    x_stars.append(x_star0)

    iterates.append(iterates0)
    for iterate in iterates0:
        iterates = restore(iterate, indices)
    tight_sets.append(tight_sets0)

    print('projection of y0 on P(f) = ' + str(x_star0))

    # Point close to first random point
    for i in range(reps):
        noise = random_point(n, (1 / (n * n)), False, seed=None)
        y1 = [round(y0[i] + noise[i], base_decimal_accuracy) for i in range(n)]
        y1.sort(reverse=True)
        print('y1 = ' + str(y1))
        y.append(y1)

        proj = IncFix(g, y1)
        x_star1, iterates1, tight_sets1 = proj.projection()
        x_star1 = round_list(x_star1, base_decimal_accuracy)
        print('projection of y1 on P(f) = ' + str(x_star1))

        x_stars.append(x_star1)
        iterates.append(iterates1)
        tight_sets.append(tight_sets1)

    for i in range(reps):
        plt.title('Size of tight sets with time - ' + str(reps + 1) + ' close points')
        color_mapping = {0: 'r', 1: 'b', 2: 'm', 3: 'y', 4: 'k'}
        plot_tight_sets(tight_sets[i], color_mapping[i % 5])

    plt.savefig('figures\\close-points\\' + str(n) + '-' + str(reps + 1) + '-close-points-tight-sets.png')
    plt.clf()


def random_points_unsorted(n: int, reps: int = 5):
    np.random.seed()
    g = generate_concave_function()
    print('g = ' + str(g))

    y, iterates, tight_sets, x_stars = [], [], [], []

    # Random points
    for i in range(reps):
        y0 = random_point(n, 3, True, seed=None)
        y0 = [round(i, base_decimal_accuracy) for i in y0]
        y0.sort(reverse=True)
        print('y = ' + str(y0))
        y.append(y0)

        proj = IncFix(g, y0)
        x_star0, iterates0, tight_sets0 = proj.projection()
        x_star0 = round_list(x_star0, base_decimal_accuracy)
        print('Iteration ' + str(i + 1) + ': projection of y on P(f) = ' + str(x_star0))

        x_stars.append(x_star0)
        iterates.append(iterates0)
        tight_sets.append(tight_sets0)

    for i in range(reps):
        plt.title('Size of tight sets with time - ' + str(reps) + ' random points')
        color_mapping = {0: 'r', 1: 'b', 2: 'm', 3: 'y', 4: 'k'}
        plot_tight_sets(tight_sets[i], color_mapping[i % 5])

    plt.savefig('figures\\random-points\\' + str(n) + '-' + str(reps) + '-random-points-tight-sets.png')
    plt.clf()


def close_points_absolute_match(n: int, reps: int = 40):
    np.random.seed(seeds[0])
    g = generate_concave_function(n)

    # Central random point
    y = random_point(n, std_dev_point, True, seed=None)

    proj = IncFix(g, y)
    mapping = proj.mapping
    x_star, iterates, tight_sets = proj.projection()

    tight_sets_for_multiple_points = []
    iterates, y1, x_star, tight_sets = map_items(iterates, y, x_star, tight_sets, mapping)
    print(y1)

    for i in range(reps):
        print('Rep ' + str(i))
        noise = random_point(n, (1/(n*n)), False, seed=None)
        y0 = [round(abs(y[i] + noise[i]), base_decimal_accuracy) for i in range(n)]

        proj0 = IncFix(g, y0)
        x_star0, iterates0, tight_sets0 = proj0.projection()
        iterates0, y0, x_star0, tight_sets0 = map_items(iterates0, y0, x_star0, tight_sets0, mapping)
        tight_sets_for_multiple_points.append(tight_sets0)

    tight_set_accuracy = []
    t, T = 0, 0
    for iteration in range(len(tight_sets)):
        tight_set1 = tight_sets[iteration]
        for rep in range(reps):
            if len(tight_sets_for_multiple_points[rep]) > iteration:
                tight_set2 = tight_sets_for_multiple_points[rep][iteration]
                if len(tight_set1) > 0 or len(tight_set2) > 0:
                    if tight_set1 == tight_set2:
                        t += 1
                    T += 1

        if T > 0:
            tight_set_accuracy.append(round(t/T, base_decimal_accuracy))

    return tight_set_accuracy


def random_points_absolute_match(n: int, reps: int = 40):
    np.random.seed(seeds[0])
    g = generate_concave_function(n)

    # Central random point
    y = random_point(n, std_dev_point, True, seed=None)

    proj = IncFix(g, y)
    mapping = proj.mapping
    x_star, iterates, tight_sets = proj.projection()

    tight_sets_for_multiple_points = []
    iterates, y, x_star, tight_sets = map_items(iterates, y, x_star, tight_sets, invert(mapping))

    for i in range(reps):
        print('Rep ' + str(i))
        y0 = random_point(n, std_dev_point, True, seed=None)

        proj0 = IncFix(g, y0)
        x_star0, iterates0, tight_sets0 = proj0.projection()
        iterates0, y0, x_star0, tight_sets0 = map_items(iterates0, y0, x_star0, tight_sets0, invert(mapping))
        tight_sets_for_multiple_points.append(tight_sets0)

    tight_set_accuracy = []
    t, T = 0, 0
    for iteration in range(len(tight_sets)):
        tight_set1 = tight_sets[iteration]
        for rep in range(reps):
            if len(tight_sets_for_multiple_points[rep]) > iteration:
                tight_set2 = tight_sets_for_multiple_points[rep][iteration]
                if len(tight_set1) > 0 or len(tight_set2) > 0:
                    if tight_set1 == tight_set2:
                        t += 1
                    T += 1

        if T > 0:
            tight_set_accuracy.append(round(t/T, base_decimal_accuracy))

    return tight_set_accuracy


def close_points_repeated_tight_sets(n: int, reps: 100, seed):
    """
    :param n:
    :param reps:
    :return:
    """
    np.random.seed(seeds[seed])
    g = list(np.ones(n))

    # Central random point
    y = list(np.ones(n))

    proj = IncFix(g, y)
    mapping = proj.mapping
    x_star, iterates, tight_sets = proj.projection()

    tight_sets_master_set = set([])
    iterates, y1, x_star, tight_sets = map_items(iterates, y, x_star, tight_sets, mapping)
    # print(tight_sets)
    tight_sets = set(frozenset(i) for i in tight_sets)
    all_tight_sets = [tight_sets]
    tight_sets_master_set = tight_sets_master_set.union(tight_sets)
    already_seen_tight_sets = [len(tight_sets)]

    # print(y1)

    for i in range(reps):
        print('Rep ' + str(i))
        noise = random_point(n, (1 / math.sqrt(n)), False, seed=None)
        y0 = [round(abs(y[i] + noise[i]), base_decimal_accuracy) for i in range(n)]

        proj0 = IncFix(g, y0)
        x_star0, iterates0, tight_sets0 = proj0.projection()
        iterates0, y0, x_star0, tight_sets0 = map_items(iterates0, y0, x_star0, tight_sets0, mapping)
        tight_sets0 = set(frozenset(i) for i in tight_sets0)
        tight_sets0 = set(tight_sets0)
        all_tight_sets.append(tight_sets0)
        already_seen_tight_sets.append(len(tight_sets0 - tight_sets_master_set))
        tight_sets_master_set = tight_sets_master_set.union(tight_sets0)

    T = len(tight_sets_master_set)
    cumulative_tight_sets_seen = [already_seen_tight_sets[0]]
    for i in range(1, len(already_seen_tight_sets)):
        cumulative_tight_sets_seen.append(cumulative_tight_sets_seen[-1] + already_seen_tight_sets[i])

    # print(cumulative_tight_sets_seen)

    already_seen_tight_sets_fraction = [cumulative_tight_sets_seen[rep]/T for rep in range(len(cumulative_tight_sets_seen))]

    return all_tight_sets
    # return already_seen_tight_sets_fraction


def close_points_metric(n: int, reps: int = 19):
    np.random.seed(seeds[0])
    g = generate_concave_function(n)
    card_pol = CardinalityPolytope(g)

    metrics = []

    # Central random point
    y = random_point(n, std_dev_point, True, seed=None)
    metrics.append(l_norm_violated_constraints(1, card_pol, y))

    for i in range(reps):
        # print('Rep ' + str(i))
        noise = random_point(n, (1/(n*n)), False, seed=None)
        y0 = [round(abs(y[i] + noise[i]), base_decimal_accuracy) for i in range(n)]
        metrics.append(l_norm_violated_constraints(1, card_pol, y0))

    metrics = round_list(metrics, base_decimal_accuracy)
    return metrics, statistics.stdev(metrics)


def random_points_metric(n: int, reps: int = 20):
    np.random.seed(seeds[0])
    g = generate_concave_function(n)
    card_pol = CardinalityPolytope(g)

    metrics = []

    for i in range(reps):
        # print('Rep ' + str(i))
        y0 = random_point(n, std_dev_point, True, seed=None)
        metrics.append(l_norm_violated_constraints(1, card_pol, y0))

    metrics = round_list(metrics, base_decimal_accuracy)
    return metrics, statistics.stdev(metrics)


# for n in [25, 50, 75]:
#     print('n = ' + str(n))
#     t1 = close_points(n)
#     t2 = random_points(n)
#
#     plt.plot(t1, 'r')
#     plt.plot(t2, 'b')
#     plt.xlabel('Number of iterations')
#     plt.ylabel('Fraction of tight sets that \'match\' (cumulative)')
#     plt.savefig('figures\\tight-set-matches\\' + str(n))
#     plt.clf()


# inner_reps = 50
# outer_reps = 50
#
#
# for n in [50, 100]:
#     master_seen_tight_sets = [0] * (inner_reps + 1)
#     for i in range(outer_reps):
#         print('Iteration ' + str(i))
#         seen_tight_sets = np.array(close_points_repeated_tight_sets(n, inner_reps, i))
#         # print(seen_tight_sets)
#         for j in range(inner_reps + 1):
#             master_seen_tight_sets[j] = master_seen_tight_sets[j] + seen_tight_sets[j]/outer_reps
#
#     # t = np.arange(0., 5., 0.2)
#     # plt.plot(t, t / inner_reps, 'r')
#     plt.title('Fraction of tight sets seen vs number of points for noise 1/sqrt(n). n = ' + str(n))
#     plt.xlabel('Number of points')
#     plt.ylabel('Fraction of total number of tight sets already seen')
#     plt.plot(master_seen_tight_sets, 'b')
#     plt.savefig('figures\\n=' + str(n) + '-fraction-of-tight-sets-seen-noise-1-by-sqrt-n.png')
#     plt.clf()
#
#
# stdev_list_1 = []
# stdev_list_2 = []
# stdev_ratio = []
# for n in range(1, 19):
#     print(n)
#     _, stdev_1 = close_points_metric(n)
#     _, stdev_2 = random_points_metric(n)
#     print(stdev_2 / stdev_1)
#     stdev_list_1.append(stdev_1)
#     stdev_list_2.append(stdev_2)
#     stdev_ratio.append(stdev_2 / stdev_1)
#
# print(round_list(stdev_list_1, base_decimal_accuracy))
# print(round_list(stdev_list_2, base_decimal_accuracy))
# print(round_list(stdev_ratio, base_decimal_accuracy))
#
# plt.plot(stdev_list_1)
# plt.show()
# plt.clf()
# plt.plot(stdev_list_2)
# plt.show()
# plt.clf()
# plt.plot(stdev_ratio)
# plt.show()
# plt.clf()


t = close_points_repeated_tight_sets(100, 100, seed=0)
print(t)
# plt.plot(t)
# plt.show()

