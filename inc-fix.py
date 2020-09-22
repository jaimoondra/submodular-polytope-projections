from typing import List
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


class CardinalityPolytope:
    def __init__(self, g: List[float]):
        if not self.is_cardianlity_submodular([0] + g):
            raise TypeError('The tuple g does not induce a cardinality based polytope.')
        self.g = [0] + g

    def __len__(self):
        return len(self.g) - 1

    @staticmethod
    def is_cardianlity_submodular(g: List[float]):
        n = len(g)

        for i in range(n):
            for j in range(n - i):
                if g[i] + g[j] < g[i + j]:
                    return False

        return True

    def is_feasible(self, x: List[float]):
        y = x.sort(reverse=True)

        n = len(x)
        if n != len(self):
            raise ValueError('The dimension ' + str(n) + ' of the point does not match the dimension ' + str(len(
                self)) + ' of the ground set.')

        prefix_sum_x = 0
        for i in range(n):
            prefix_sum_x += x[i]
            if prefix_sum_x > self.g[i + 1]:
                return False

        return True

    def is_feasible_in_base(self, x: List[float]):
        return True if sum(x) == self.g[-1] and self.is_feasible(x) else False


class IncFix:
    def __init__(self, g: List[float], y: List[float]):
        self.cardinality_polytope = CardinalityPolytope(g)
        if len(self.cardinality_polytope) != len(y):
            raise ValueError('The dimension ' + str(len(y)) + ' of the point does not match the dimension ' + str(
                len(self.cardinality_polytope)) + ' of the ground set.')
        self.y = y

    def __len__(self):
        return len(self.cardinality_polytope)

    def gradient(self, x: List[float], evaluation_set: set):
        if evaluation_set is None:
            evaluation_set = set(range(len(self)))
        gradient = [x[i] - self.y[i] if i in evaluation_set else np.inf for i in range(len(self))]
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

        args = list(np.argsort(x))
        args.reverse()
        y = x
        y.sort(reverse=True)
        logging.debug('y = ' + str(y))

        tight_set = set([])
        prefix_sum = 0.0
        for i in range(len(self)):
            prefix_sum += y[i]
            if prefix_sum == g[i + 1]:
                tight_set.add(args[i])
            else:
                break

        logging.debug('Tight set = ' + str(tight_set))
        return tight_set

    def projection(self):
        n = len(self)
        N = set(range(n))
        x, i = [0.0] * n, 0
        iterates = [x]
        count = 0
        while count < 4:
            i += 1
            M = self.argmin(self.gradient(x, N))
            logging.debug('Minimum gradients = ' + str(M))
            logging.debug('Maximal tight set = ' + str(self.maximal_tight_set(x)))

            count2 = 0
            while self.maximal_tight_set(x).intersection(M) == set([]) and count2 < 5:
                count2 += 1
                shift_1 = min(self.gradient(x, N - M)) - min(self.gradient(x, N))
                shift_2 = self.max_feasible_shift(x, M)
                logging.debug('Shifts = ' + str(shift_1) + ', ' + str(shift_2))

                x = self.shift_on_set(x, M, min(shift_1, shift_2))
                logging.debug('x = ' + str(x))

                iterates.append(x)
                M = self.argmin(self.gradient(x, N))

            N = N - (M.intersection(self.maximal_tight_set(x)))
            if len(N) == 0:
                break

            count += 1

        return x, iterates

    @staticmethod
    def shift_on_set(x: List[float], M: set, shift: float):
        for i in range(len(x)):
            if i in M:
                x[i] += shift

        return x

    def max_feasible_shift(self, x: List[float], M: set):
        logging.info('Determining maximum feasible set.')

        prefix_sum = self.prefix_sum(x)
        logging.debug('prefix_sum = ' + str(prefix_sum))

        diff = []
        logging.debug('M = ' + str(M))

        for i in range(len(x)):
            intersect = len(M.intersection(set(range(i + 1))))
            logging.debug('i, intersect = ' + str(i) + ', ' + str(intersect))
            if intersect == 0:
                diff.append(np.inf)
            else:
                diff.append((self.cardinality_polytope.g[intersect + 1] - prefix_sum[i])/intersect)
        return min(diff)

    @staticmethod
    def prefix_sum(x: List[float]):
        prefix_sum = [x[0]]
        for i in range(1, len(x)):
            prefix_sum.append(prefix_sum[i - 1] + x[i])

        return prefix_sum


g = [0.4, 0.6, 0.7]
y = [0.06, 0.07, 0.6]
proj = IncFix(g, y)
print(proj.projection())
