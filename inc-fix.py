from typing import List
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

minimum_decimal_difference = 0.0000001


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
                return False

        # Check if g is concave
        for i in range(n - 1):
            if g[i] + g[i + 2] > 2 * g[i + 1]:
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
            prefix_sum.append(round(prefix_sum[i - 1] + x[i], 5))

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
        y.sort(reverse=True)
        self.y = y

    def __len__(self):
        return len(self.cardinality_polytope)

    def gradient(self, x: List[float], evaluation_set: set = None):
        if evaluation_set is None:
            evaluation_set = set(range(len(self)))

        gradient = [round(x[i] - self.y[i], 5) if i in evaluation_set else np.inf for i in range(len(self))]
        logging.debug('Gradient for x = ' + str(x) + ' is ' + str(gradient))

        return gradient

    @staticmethod
    def argmin(numbers: List[float]):
        min_value, args = np.min(numbers), set([])
        for i in range(len(numbers)):
            if numbers[i] == min_value:
                args.add(i)

        return args

    def maximal_tight_set(self, x: List[float]):
        logging.info('Determining maximal tight set.')
        logging.debug('x = ' + str(x))

        prefix_sum = self.cardinality_polytope.prefix_sum(x)
        tight_set = set([])
        for i in range(len(self)):
            if prefix_sum[i] == self.cardinality_polytope.g[i + 1]:
                tight_set = set(range(i + 1))

        logging.debug('Tight set = ' + str(tight_set))
        return tight_set

    def projection(self):
        n = len(self)
        N = set(range(n))
        x, i = [0.0] * n, 0
        iterates = [x]
        fixed = set([])
        count = 0

        while True:
            i += 1
            M = self.argmin(self.gradient(x, N))
            logging.debug('Minimum gradients = ' + str(M))

            count2 = 0
            while self.maximal_tight_set(x).intersection(M) == set([]):
                count2 += 1
                shift_1 = min(self.gradient(x, N - M)) - min(self.gradient(x, N))
                shift_2 = self.max_feasible_shift(x, M)
                logging.debug('Shifts = ' + str(shift_1) + ', ' + str(shift_2))

                x = self.shift_on_set(x, M, min(shift_1, shift_2))
                logging.debug('x = ' + str(x))

                iterates.append(x)
                M = self.argmin(self.gradient(x, N))

            maximal_tight_set = self.maximal_tight_set(x)
            fixed = fixed.union(M.intersection(maximal_tight_set))
            logging.debug('Fixed set = ' + str(fixed))

            N = N - (M.intersection(maximal_tight_set))
            if len(N) == 0:
                break

            count += 1

        return x, iterates

    @staticmethod
    def shift_on_set(x: List[float], M: set, shift: float):
        shifted_x = [round(x[i] + shift, 5) if i in M else x[i] for i in range(len(x))]
        return shifted_x

    def max_feasible_shift(self, x: List[float], M: set):
        logging.info('Determining maximum feasible set.')

        prefix_sum_x = self.cardinality_polytope.prefix_sum(x)
        logging.debug('prefix_sum = ' + str(prefix_sum_x))

        k = min(M)
        shifts = [round((self.cardinality_polytope.g[i + 1] - prefix_sum_x[i])/(i + 1 - k), 5) for i in range(k,
                                                                                                          k + len(M))]
        shift = min(shifts)
        return shift


g = [0.4, 0.6, 0.7, 0.8]
y = [0.05, 0.07, 1, 0.6]
proj = IncFix(g, y)
x_star, iterates = proj.projection()
print(x_star, iterates)
