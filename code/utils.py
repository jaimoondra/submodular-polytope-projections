import math
import numpy as np
import random
from typing import List, Dict, Set
from constants import *
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd


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


def determine_tight_sets(y, x, g=None):
    """
    Given a point y and its projection x, determines the tight sets
    :param y: projected point, np array
    :param x: projection, np array
    :return: sequence of tight cuts alogn with c[j + 1] - c[j] value, sorted in decreasing order
    of values of c[j + 1] - c[j]
    """
    # def set_sum(S: Set):
    #     x_s = 0.0
    #     for s in S:
    #         x_s = x_s + x[s]
    #     return x_s
    #
    # def get_gradients(S: set):
    #     w = {s: round(x[s] - y[s], 4) for s in S}
    #     return w

    n = len(x)
    z = x - y
    z1, mapping = sort(list(z))
    inverse_mapping = invert(mapping)

    tight_sets = []

    flag = np.zeros(len(z1))
    for i in range(1, len(z1)):
        if abs(z1[i] - z1[i - 1]) >= 0.000001:
            flag[i] = 1

    H, F = set(), set()
    for i in range(n):
        if flag[i] == 1:
            tight_sets.append([z1[i] - z1[i - 1], frozenset(H)])
            F = {inverse_mapping[i]}
        else:
            F = F.union({inverse_mapping[i]})
        H = H.union({inverse_mapping[i]})

    tight_sets.append([np.inf, frozenset(H)])
    # unique_gradient_coordinates = np.unique(z)
    # unique_gradient_coordinates = np.append(unique_gradient_coordinates, np.inf)
    # for j in range(len(unique_gradient_coordinates) - 1):
    #     F = set(np.where(z1 == unique_gradient_coordinates[j])[0])
    #     T = T.union(F)
    #     # print(T, set_sum(T))
    #     # print(get_gradients(T))
    #     tight_sets.append([round(unique_gradient_coordinates[j + 1] -
    #                          unique_gradient_coordinates[j], 8), frozenset(T)])

    tight_sets = np.array(tight_sets)
    tight_sets = tight_sets[np.argsort(tight_sets[:, 0])]

    return tight_sets[::-1]


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


def random_point(n: int, mean: float = 0, r: float = 1, nonnegative: bool = False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # z = np.random.randint(1, n, n)
    z = np.random.multivariate_normal(mean=[mean] * n, cov=(r * r) * np.identity(n))
    if nonnegative:
        z = [abs(round(x, 2)) for x in z]
    else:
        z = [round(x, 2) for x in z]

    return z


def random_error(n: int, r: float = 1, seed=None, decial_accuracy: int = base_decimal_accuracy):
    """
    Generates random error vector in dimension n, where each coordinate is from a Gaussian
    distribiution with mean 0 and standard deviation r
    :param n: dimension
    :param r: standrad deviation in each coordinate
    :param seed: random seed
    :param decial_accuracy: number of digists after the decimal point
    :return:
    """
    if seed is not None:
        np.random.seed(seed)

    z = np.random.multivariate_normal(mean=[0] * n, cov=(r * r) * np.identity(n))
    return [round(x, decial_accuracy) for x in z]


# print(determine_tight_sets(np.array([0, 0, 0, 0, 0]), np.array([1, 3, 1, 0, 6])))
