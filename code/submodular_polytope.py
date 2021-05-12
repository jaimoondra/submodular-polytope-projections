from typing import List, Dict, Set
import numpy as np
import math
import logging
from utils import *
from constants import *
import networkx as nx
from networkx.algorithms import bipartite


logging.basicConfig(level=logging.INFO)


class SubmodularFunction:
    def __init__(self, n: int = 0):
        """
        :param S: ground set; if you want to make the ground set [1, ..., n] set S = {} and see n
        :param n: if you want the ground set to be the range [1, ..., n]
        """
        self.n = n

    def __len__(self):
        return self.n

    def function_value(self, T: set):
        if not T.issubset(range(self.n)):
            raise ValueError("The provided set is not a subset of the ground set.")
        return 0.0


class CardinalityDifferenceSubmodularFunction(SubmodularFunction):
    def __init__(self, g: List[float], r: List[float], n: int = 0):
        super().__init__(n)
        # We assume that the list g is the tuple (g(1), ..., g(n)). We append g(0) = 0 to this list
        # Check if g induces a submodular function f
        if not self.is_cardianality_submodular([0.0] + g):
            raise TypeError('The tuple g does not induce a cardinality based polytope.')

        if len(g) != len(r) or len(g) != n:
            print(n, len(g), len(r))
            raise ValueError('Sizes not same.')

        self.g = [0.0] + g
        self.r = r

    def function_value(self, T: set):
        if not T.issubset(range(len(self) + 1)):
            raise ValueError("The provided set is not a subset of the ground set.")

        return self.g[len(T)]  # - r_T

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
                logging.debug('Nonmonotonic: i = ' + str(i) + ', g[i] = ' + str(g[i]) +
                              ', g[i + 1] = ' + str(g[i + 1]))
                return False

        # Check if g is concave
        for i in range(n - 1):
            if g[i] + g[i + 2] > 2 * g[i + 1] + minimum_decimal_difference:
                logging.debug('Not concave: i = ' + str(i) + ', g[i] = ' + str(g[i]) +
                              ', g[i + 1] = ' + str(g[i + 1]) + ', g[i + 2] = ' + str(g[i + 2]))
                return False

        return True


class CardinalitySubmodularFunction(CardinalityDifferenceSubmodularFunction):
    def __init__(self, g: List[float], n):
        r = [0.0] * n
        super().__init__(g, r, n)


class PermutahedronSubmodularFunction(CardinalitySubmodularFunction):
    def __init__(self, n: int = 1):
        g = [float(n)]
        for i in range(1, n):
            g.append(float(g[-1] + n - i))
        super().__init__(g, n)


class SubmodularPolytope:
    def __init__(self, f: SubmodularFunction):
        self.f = f

    def __len__(self):
        return len(self.f)

    def is_feasible(self, x):
        pass

    def linear_optimization_over_base(self, c: List[float]):
        """
        Returns max (c^T x) and argmin (c^T x) for x in B(f)
        :param c: Vector of length |E|
        :return: max c^T x for x in B(f)
        """
        d, mapping = sort(c, reverse=True)  # Sort in revese order for greedy algorithm
        x = [0.0] * len(self)  # This will be the argmax
        # Greedy algorithm
        for i in range(len(self)):
            x[i] = self.f.function_value(set(range(i + 1))) - self.f.function_value(set(range(i)))

        opt = sum([x[i] * d[i] for i in range(len(self))])  # Opt value
        invert_x = permute(x, invert(mapping))  # Restore the original order for x

        return opt, invert_x

    def linear_optimization_tight_sets(self, c: List[float], T: List[Set]):
        """
        Linear optimization over B(f) with additional constraints. Every set in T should also be
        tight. Returns max (c^Tx) and argmin (c^T x) for x in B(f), with the additional
        constraints that x(U) = f(U) for all U in T.
        :param c: cost vector
        :param T: set of tight sets. Assumed to be a chain for now, and T is assumed to be
        increasing. T[0] is assumed to be the emptyset set({}) and T[-1] is assumed to be the
        ground set
        :return: max c^T x under the constraints
        """
        permutation = {}
        count = 0
        for j in range(1, len(T)):
            U = T[j]
            D = U.difference(T[j - 1])
            for u in D:
                permutation[u] = count
                count = count + 1

        c1 = permute(c, permutation)
        x = []
        l = 0
        mappings = {}
        opt = 0

        for j in range(1, len(T)):
            U = T[j]
            y = []
            D = U.difference(T[j - 1])
            m = len(D)
            d, mapping = sort(c1[l: l + m], reverse=True)
            inverse_mapping = invert(mapping)
            for i in range(m):
                y.append(self.f.function_value(set(range(i + l + 1))) - self.f.function_value(set(
                    range(l + i))))
                opt = opt + d[i] * (self.f.function_value(set(range(l + i + 1))) -
                                    self.f.function_value(set(range(l + i))))
            y = permute(y, inverse_mapping)
            x = x + y
            l = l + m

        z = permute(x, invert(permutation))
        return opt, z

    def affine_minimizer(self, S: List[List[float]]):
        n = len(self)
        B = np.column_stack(S)
        C = np.transpose(B)
        D = np.linalg.inv(np.matmul(C, B))
        o = np.ones(n)
        alpha = (np.matmul(D, o)) / (np.matmul(o, np.matmul(D, o)))
        y = np.matmul(B, alpha)
        return y, alpha

    def minimum_norm_point(self, eps: float):
        def get_base_vertex():
            return [self.f.function_value(set(range(i + 1))) - self.f.function_value(set(range(i)))
                    for i in range(n)]

        def nonnegative_coordinates(y: List[float]):
            C = {}
            for i in range(len(y)):
                if y[i] < 0:
                    C.update({i: y[i]})

            return C

        eps = abs(eps)
        x = get_base_vertex()
        S = [x]  # Set S in the algorithm
        L = [1]  # Coefficients lambda_i
        s = 1  # |S|
        while True:
            _, q = self.linear_optimization_over_base(x)
            if np.linalg.norm(x) * np.linalg.norm(x) <= np.matmul(x, q) + eps * eps:
                break

            S = S + [q]
            L.append(0.0)

            while True:
                y, alpha = self.affine_minimizer(S)
                C = nonnegative_coordinates(alpha)
                if len(C) == 0:
                    break

                theta = min([L[k] / (L[k] - alpha[k]) for k in C])
                x = theta * y + (1 - theta) * x
                L = theta * y + (1 - theta) * L

            x = y

        return x


class CardinalityPolytope(SubmodularPolytope):
    """
    Class for cardinality based polytopes. Let N be the ground set of size n. Then,
    f is a submodular function on the power set of N, given by f(A) = g(|A|) for each subset A of
    N. g is called the cardinality function.
    """

    def __init__(self, f: CardinalitySubmodularFunction):
        super().__init__(f)
        self.f = f

    def __len__(self):
        """
        :return: size of the ground set S
        """
        return len(self.f)

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
            raise ValueError('The dimension ' + str(n) + 'of the point does not match the '
                                                         'dimension ' + str(
                len(self)) + ' of the ground set.')

        # Check if x[0] + x[1] + ... + x[i - 1] <= g[i]
        prefix_sum_x = self.prefix_sum(x)
        for i in range(n):
            if prefix_sum_x[i] > self.f.g[i + 1]:
                return False

        return True

    def is_feasible_in_base(self, x: List[float]):
        """
        Checks if x in is B(f)
        :param x: point in space
        :return: True if x is in B(f), False otherwise
        """
        return True if sum(x) == self.f.g[-1] and self.is_feasible(x) else False

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


class Permutahedron(CardinalityPolytope):
    def __init__(self, f: PermutahedronSubmodularFunction):
        super().__init__(f)


class BiparititeGraphSubmodularFunction(SubmodularFunction):
    def __init__(self, n: int, G: nx.Graph, U, V):
        super().__init__(n)
        self.G = G
        self.U = U
        self.V = V

    def function_value(self, T: set):
        """
        Get f(T) = w(N(T)), the total weight of neighborhoods of T
        :param T: subset of V
        :return: f(T) = w(N(T))
        """
        w = 0
        N = set()
        for t in T:
            for u in self.G.neighbors(t):
                if u not in N:
                    N.add(u)
                    w = w + self.G.get_edge_data(u, t)['weight']

        return w


def generate_random_bipartite_graph(n: int):
    for i in range(n):
        for j in range(n + 1, 2*n):
             p = random.random()
             q = random.random()
             if q <= p:
                 w = random.random()
