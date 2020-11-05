from typing import List, Dict, Set
import numpy as np
import logging
from constants import *


logging.basicConfig(level=logging.WARNING)


def sort(x: List[float], reverse=False):
    """
    :param x: List of numbers
    :param reverse: Sorts in decreasing order if set to True
    :return: Sorted list and the corresponding sorted indices
    """
    j = 0
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


def permute(x: List[float], mapping: Dict[int, int]):
    y = [0] * len(x)
    for i in range(len(x)):
        y[mapping[i]] = x[i]

    return y


class CardinalityPolytope:
    """
    Class for cardinality based polytopes.
    Let N be the ground set of size n. Then, f is a submodular function on the power set of N, given by f(A) = g(|A|)
    for each subset A of N. g is called the cardinality function.
    """

    def __init__(self, g: List[float]):
        # We assume that the list g is the tuple (g(1), ..., g(n)). We append g(0) = 0 to this list
        # Check if g induces a submodular function f
        if not self.is_cardianality_submodular([0] + g):
            raise TypeError('The tuple g does not induce a cardinality based polytope.')

        self.g = [0] + g

    def __len__(self):
        """
        :return: size of the ground set N
        """
        return len(self.g) - 1

    @staticmethod
    def is_cardianality_submodular(g: List[float]):
        """
        Checks if the cardinality function g induces submodular f
        :param g: cardinality function g on {0, 1, ..., n}
        :return: True if f is submodular, False otherwise
        """
        n = len(g) - 1

        # Check if g is monotonic nonincreasing
        for i in range(n):
            if g[i + 1] < g[i]:
                logging.debug('Nonmonotonic: i = ' + str(i) + ', g[i] = ' + str(g[i]) + ', g[i + 1] = ' + str(g[i + 1]))
                return False

        # Check if g is concave
        for i in range(n - 1):
            if g[i] + g[i + 2] > 2 * g[i + 1] + minimum_decimal_difference:
                logging.debug('Not concave: i = ' + str(i) + ', g[i] = ' + str(g[i]) + ', g[i + 1] = ' +
                              str(g[i + 1]) + ', g[i + 2] = ' + str(g[i + 2]))
                return False

        return True

    def is_feasible(self, x: List[float]):
        """
        Checks if point x is in P(f)
        :param x: point in space
        :return: True if x is in P(f), False, otherwise
        """
        # Descending sort
        x.sort(reverse=True)

        n = len(x)
        # Check if dimensions match:
        if n != len(self):
            raise ValueError('The dimension ' + str(n) + ' of the point does not match the dimension ' + str(len(
                self)) + ' of the ground set.')

        # Check if x[0] + x[1] + ... + x[i - 1] <= g[i]
        prefix_sum_x = self.prefix_sum(x)
        for i in range(n):
            if prefix_sum_x[i] > self.g[i + 1]:
                return False

        return True

    def is_feasible_in_base(self, x: List[float]):
        """
        Checks if x in is B(f)
        :param x: point in space
        :return: True if x is in B(f), False otherwise
        """
        return True if sum(x) == self.g[-1] and self.is_feasible(x) else False

    @staticmethod
    def prefix_sum(x: List[float]):
        """
        :param x: Point in space
        :return: prefix sum of x
        """
        prefix_sum = [x[0]]
        for i in range(1, len(x)):
            prefix_sum.append(round(prefix_sum[i - 1] + x[i], high_decimal_accuracy))

        return prefix_sum

    @staticmethod
    def distance(x: float, y: float):
        return abs(x - y)


class IncFix:
    """
    Class for IncFix
    """

    def __init__(self, g: List[float], y: List[float]):
        self.cardinality_polytope = CardinalityPolytope(g)

        # Check if point y has the correct dimension
        if len(self.cardinality_polytope) != len(y):
            raise ValueError('The dimension ' + str(len(y)) + ' of the point does not match the dimension ' + str(
                len(self.cardinality_polytope)) + ' of the ground set.')

        self.y, self.mapping = sort(y, reverse=True)

    def __len__(self):
        """
        :return: Size of ground set
        """
        return len(self.cardinality_polytope)

    def gradient(self, x: List[float], evaluation_set: set = None):
        if evaluation_set is None:
            evaluation_set = set(range(len(self)))

        gradient = [round(x[i] - self.y[i], high_decimal_accuracy) if i in evaluation_set else np.inf
                    for i in range(len(self))]
        logging.debug('GRADIENT Gradient for x = ' + str(x) + ' is ' + str(gradient))

        return gradient

    @staticmethod
    def argmin(numbers: List[float]):
        """
        :param numbers: List of (real) numbers
        :return: The set of indices with minimum value
        """
        min_value, args = np.min(numbers), set([])
        for i in range(len(numbers)):
            if numbers[i] == min_value:
                args.add(i)

        return args

    def maximal_tight_set(self, x: List[float]):
        """
        :param x: Point in space
        :return: Maximal tight set with respect to P(f) - can be proven to be unique
        """
        prefix_sum = self.cardinality_polytope.prefix_sum(x)
        tight_set = set([])
        for i in range(len(self)):
            if abs(prefix_sum[i] - self.cardinality_polytope.g[i + 1]) <= 2 * minimum_decimal_difference:
                tight_set = set(range(i + 1))

        return tight_set

    def projection(self):
        """
        Implements inc-fix
        :return: projection of point y on P(f)
        """
        n = len(self)
        N = set(range(n))
        x, i = [0.0] * n, 0
        iterates = [x]
        tight_sets = []

        while True:
            i += 1
            M = self.argmin(self.gradient(x, N))
            logging.debug('INCFIX Argmin for  gradients = ' + str(M))

            while self.maximal_tight_set(x).intersection(M) == set([]):
                shift_1 = min(self.gradient(x, N - M)) - min(self.gradient(x, N))
                shift_2 = self.max_feasible_shift(x, M)
                logging.debug('INCFIX Shifts = ' + str(shift_1) + ', ' + str(shift_2))

                x = self.shift_on_set(x, M, min(shift_1, shift_2))
                logging.debug('INCFIX x = ' + str(x))
                logging.debug('INCFIX M = ' + str(M))
                logging.debug('INCFIX T(x) = ' + str(self.maximal_tight_set(x)))

                maximal_tight_set = self.maximal_tight_set(x)
                iterates.append(x)
                M = self.argmin(self.gradient(x, N))

            maximal_tight_set = self.maximal_tight_set(x)
            tight_sets.append(maximal_tight_set)
            logging.debug('INCFIX Tight set sequence = ' + str(maximal_tight_set))
            logging.info('INCFIX Tight set size = ' + str(len(maximal_tight_set)))

            N = N - (M.intersection(maximal_tight_set))
            if len(N) == 0:
                break

        x = [round(x[i], base_decimal_accuracy) for i in range(len(x))]

        return self.restore(x, iterates, tight_sets)

    def restore(self, x: List[float], iterates: List[List[float]], tight_sets: List[Set[int]]):
        inverse_mapping = invert(self.mapping)
        x = permute(x, inverse_mapping)
        for i in range(len(iterates)):
            iterates[i] = permute(iterates[i], inverse_mapping)

        for i in range(len(tight_sets)):
            tight_sets[i] = map_set(tight_sets[i], inverse_mapping)

        return x, iterates, tight_sets

    @staticmethod
    def shift_on_set(x: List[float], M: set, shift: float):
        """
        :param x: Point in space
        :param M: Subset of ground set (= {0, 1, ..., N - 1}) on which to shift
        :param shift: Amount to shift
        :return: Shifted point
        """
        shifted_x = [round(x[i] + shift, high_decimal_accuracy) if i in M else x[i] for i in range(len(x))]
        return shifted_x

    def max_feasible_shift(self, x: List[float], M: set):
        """
        :param x: Point in space
        :param M: Subset of ground set (= {0, 1, ..., N - 1}) on which
        :return: The largest number that can be added to x on the set M so that the point is still in P(f)
        """

        # A proof can be found in Swati's thesis

        prefix_sum_x = self.cardinality_polytope.prefix_sum(x)
        logging.debug('SHIFT prefix_sum = ' + str(prefix_sum_x))
        logging.debug('SHIFT g = ' + str(self.cardinality_polytope.g))

        k = min(M)
        shifts = [(self.cardinality_polytope.g[i + 1] - prefix_sum_x[i])/(i + 1 - k) for i in range(k, k + len(M))]
        logging.debug('SHIFT Shifts array = ' + str(shifts))

        shift = min(shifts)
        return shift
