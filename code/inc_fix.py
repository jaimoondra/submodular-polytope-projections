from typing import List, Dict, Set
import numpy as np
import logging
from constants import *
from submodular_polytope import CardinalityPolytope, CardinalitySubmodularFunction
from utils import *

logging.basicConfig(level=logging.WARNING)


class IncFix:
    """
    Class for IncFix
    """

    def __init__(self, g: List[float], y: List[float], x: List[float]=None):
        f = CardinalitySubmodularFunction(n=len(g), g=g)
        self.cardinality_polytope = CardinalityPolytope(f=f)

        # Check if point y has the correct dimension
        if len(self.cardinality_polytope) != len(y):
            raise ValueError('The dimension ' + str(len(y)) + 'of the point does not match the '
                             'dimension ' + str(len(self.cardinality_polytope)) + ' of the ground '
                             'set.')

        # We sort y in decreasing order because it is easy to describe our algorithm in that setting
        self.y, self.mapping = sort(y, reverse=True)
        # Inner loop iterates are also added
        self.iterates = []
        self.x = [0.0] * len(self) if x is None else x
        self.tight_sets = []
        self.gradient = self.calculate_gradient(self.x, set(range(len(self))))

    def __len__(self):
        """
        :return: Size of ground set
        """
        return len(self.cardinality_polytope)

    def get_iterates(self):
        if self.iterates is []:
            raise ValueError('INCFIX Error: Projection not run yet; iterates not determined')
        return self.iterates

    def get_tight_sets(self):
        if self.tight_sets is set():
            raise ValueError('INCFIX Error: Projection not run yet; tight_sets not determined')
        return self.tight_sets

    def get_gradient(self):
        return self.gradient

    def calculate_gradient(self, x: List[float], evaluation_set: set = None):
        if evaluation_set is None:
            evaluation_set = set(range(len(self)))

        gradient = [round(x[i] - self.y[i], high_decimal_accuracy) if i in evaluation_set
                    else np.inf for i in range(len(self))]
        logging.debug('GRADIENT Gradient for x = ' + str(x) + ' is ' + str(gradient))

        return gradient

    @staticmethod
    def argmin(numbers: List[float]):
        """
        :param numbers: List of (real) numbers
        :return: The set of indices with minimum value
        """
        min_value, args = np.min(numbers), set()
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
        tight_set = set()
        for i in range(len(self)):
            if abs(prefix_sum[i] - self.cardinality_polytope.f.g[i + 1]) <= 2 * \
                    minimum_decimal_difference:
                tight_set = set(range(i + 1))

        return tight_set

    def projection(self):
        """
        Implements inc-fix
        :return: projection of point y on P(f)
        """
        n = len(self)
        N = set(range(n))
        self.iterates = [self.x]
        self.tight_sets = [set()]

        while True:
            M = self.argmin(self.calculate_gradient(self.x, N))
            logging.debug('INCFIX Argmin for  gradients = ' + str(M))

            while self.maximal_tight_set(self.x).intersection(M) == set():
                logging.debug('INCFIX Maximal tight set = ' + str(self.maximal_tight_set(self.x)))
                logging.debug('INCFIX M = ' + str(M))
                shift_1 = min(self.calculate_gradient(self.x, N - M)) -\
                          min(self.calculate_gradient(self.x, N))
                shift_2 = self.max_feasible_shift(self.x, M)
                logging.debug('INCFIX Shifts = ' + str(shift_1) + ', ' + str(shift_2))

                self.x = self.shift_on_set(self.x, M, min(shift_1, shift_2))
                logging.debug('INCFIX x = ' + str(self.x))
                logging.debug('INCFIX M = ' + str(M))
                logging.debug('INCFIX T(x) = ' + str(self.maximal_tight_set(self.x)))

                self.iterates.append(self.x)
                M = self.argmin(self.calculate_gradient(self.x, N))

            maximal_tight_set = self.maximal_tight_set(self.x)
            self.tight_sets.append((M.intersection(maximal_tight_set)).union(self.tight_sets[-1]))
            logging.debug('INCFIX Tight set sequence = ' + str(maximal_tight_set))
            logging.info('INCFIX Tight set size = ' + str(len(maximal_tight_set)))
            self.regularize_iterate(self.x, self.tight_sets)

            N = N - (M.intersection(maximal_tight_set))
            if len(N) == 0:
                break

        self.x = [round(self.x[i], base_decimal_accuracy) for i in range(len(self.x))]
        self.gradient = self.regularize_gradient(self.x)
        self.x = self.regularize_iterate(self.x, self.tight_sets)

        # Restore the original order (see init)
        self.x, self.gradient, self.iterates, self.tight_sets = self.restore(self.x, self.iterates,
                                                              self.tight_sets, self.gradient)

        return

    def regularize_gradient(self, x: List[float]):
        """
        Remove numerical inaccuracies in gradient on set N
        :param x:
        :return:
        """
        gradient = self.calculate_gradient(x)
        for i in range(len(self) - 1):
            if gradient[i + 1] - gradient[i] < 2 * minimum_decimal_difference:
                gradient[i + 1] = gradient[i]

        return gradient

    def regularize_iterate(self, x: List[float], tight_sets: List[Set[int]]):
        """

        :param x:
        :param tight_sets:
        :return:
        """
        for i in range(1, len(tight_sets)):
            N = tight_sets[i] - tight_sets[i - 1]
            gradient = self.calculate_gradient(x, N)
            regularized = min(gradient)
            logging.debug('REGULARIZE ITERATE: regularization parameter = ' + str(regularized))
            for j in N:
                if x[j] - self.y[j] > regularized + 2 * minimum_decimal_difference:
                    raise ValueError('Gradients differ significantly for set ' + str(N) + ' for '
                                     'point ' + str({i: x[i] for i in range(len(self))}) + '\n' +
                                     'The gradient is ' + str({j: x[j] - self.y[j] for j in range(
                                     len(self))}))
                else:
                    x[j] = self.y[j] + regularized

        return x

    def restore(self, x: List[float], iterates: List[List[float]], tight_sets: List[Set[int]],
                gradient: List[float]):
        inverse_mapping = invert(self.mapping)
        x = permute(x, inverse_mapping)
        gradient = permute(gradient, inverse_mapping)

        for i in range(len(iterates)):
            iterates[i] = permute(iterates[i], inverse_mapping)

        for i in range(len(tight_sets)):
            tight_sets[i] = map_set(tight_sets[i], inverse_mapping)

        return x, gradient, iterates, tight_sets

    @staticmethod
    def shift_on_set(x: List[float], M: set, shift: float):
        """
        :param x: Point in space
        :param M: Subset of ground set (= {0, 1, ..., N - 1}) on which to shift
        :param shift: Amount to shift
        :return: Shifted point
        """
        shifted_x = [round(x[i] + shift, high_decimal_accuracy) if i in M else x[i] for i in
                     range(len(x))]
        return shifted_x

    def max_feasible_shift(self, x: List[float], M: set):
        """
        :param x: Point in space
        :param M: Subset of ground set (= {0, 1, ..., N - 1}) on which
        :return: The largest number that can be added to x on the set M so that the point is still
        in P(f)
        """
        # A proof can be found in Swati's thesis

        prefix_sum_x = self.cardinality_polytope.prefix_sum(x)
        logging.debug('SHIFT prefix_sum = ' + str(prefix_sum_x))
        logging.debug('SHIFT g = ' + str(self.cardinality_polytope.f.g))

        k = min(M)
        shifts = [(self.cardinality_polytope.f.g[i + 1] - prefix_sum_x[i]) / (i + 1 - k) for i in
                  range(k, k + len(M))]
        logging.debug('SHIFT Shifts array = ' + str(shifts))

        shift = round(min(shifts), high_decimal_accuracy)
        return shift

