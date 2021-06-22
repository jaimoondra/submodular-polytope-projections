from typing import List, Dict, Set
import numpy as np
import math
import logging

logging.basicConfig(level=logging.WARNING)

from utils import *
from constants import *
import networkx as nx
from networkx.algorithms import bipartite
from continuous_methods import AFW


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


class DifferenceSubmodularFunction(SubmodularFunction):
    """
    class to represent submodular function f'(T) = f(T) - r(T), where f is a submodular function
    and r is a vector
    """
    def __init__(self, f: SubmodularFunction, r: List[float]):
        super().__init__()
        self.f_0 = f
        self.r = r
        self.n = len(r)

    def function_value(self, T: set):
        v = self.f_0.function_value(T)
        s = sum([self.r[a] for a in T])
        return v - s


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
        inverse = invert(mapping)           # Inverse permutation

        x = [0.0] * len(self)  # This will be the argmax
        # Greedy algorithm
        for i in range(len(self)):
            S_up = map_set(set(range(i + 1)), inverse)
            S_down = map_set(set(range(i)), inverse)
            x[i] = self.f.function_value(S_up) - self.f.function_value(S_down)

        opt = sum([x[i] * d[i] for i in range(len(self))])  # Opt value
        invert_x = permute(x, inverse)  # Restore the original order for x

        return opt, invert_x

    @staticmethod
    def check_chain(T: List[Set]):
        """
        Checks if a given list of sets forms a chain
        :param T: List of sets
        :return: Boolean
        """
        flag = True
        for i in range(len(T) - 1):
            if not T[i].issubset(T[i + 1]):
                flag = False

        return flag

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
        if T[-1] is not frozenset(range(len(self))):
            T.append(frozenset(range(len(self))))

        if not self.check_chain(T):
            raise ValueError('The given chain of sets does not form a chain.')

        permutation = {}
        count = 0
        for j in range(1, len(T)):
            U = T[j]
            D = U.difference(T[j - 1])
            for u in D:
                permutation[u] = count
                count = count + 1

        inverse_permutation = invert(permutation)
        c1 = permute(c, permutation)
        x = []
        l = 0
        opt = 0

        # print(permutation, inverse_permutation)

        for j in range(1, len(T)):
            U = T[j]
            y = []
            D = U.difference(T[j - 1])
            m = len(D)
            d, mapping = sort(c1[l: l + m], reverse=True)
            inverse = invert(mapping)
            for i in range(m):
                D_up_inverse = {t + len(T[j - 1]) for t in map_set(set(range(i + 1)), inverse)}
                # D_up_inverse = set([a + l for a in list(range(i + 1))])

                D_down_inverse = {t + len(T[j - 1]) for t in map_set(set(range(i)), inverse)}
                # D_down_inverse = set([a + l for a in list(range(i))])

                S_up = T[j - 1].union(map_set(D_up_inverse, inverse_permutation))
                S_down = T[j - 1].union(map_set(D_down_inverse, inverse_permutation))

                # print('U sets', S_up, S_down)

                y.append(self.f.function_value(S_up) - self.f.function_value(S_down))
                opt = opt + d[i] * y[-1]
            y = permute(y, inverse)
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
            v = [self.f.function_value(set(range(i + 1))) - self.f.function_value(set(range(i)))
                    for i in range(n)]
            return v

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
            _, q = self.linear_optimization_over_base(-np.array(x))
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

    def minimum_norm_point_using_afw(self, eps: float):
        h = lambda x: 0.5 * np.dot(x, x)
        grad_h = lambda x: x
        h_tol, time_tol = -1, np.inf

        def lmo(x):
            _, v = self.linear_optimization_over_base(x)
            return np.array(v)

        y = np.zeros(len(self))
        x0 = lmo(y)
        # print('x0 =', x0)

        x, _, _, _, _, _ = AFW(x0, {tuple(x0): 1}, lmo, eps, h, grad_h, h_tol, time_tol)
        # print('x = ', x)
        return x

    def maximal_minimizer(self, eps):
        x = self.minimum_norm_point_using_afw(eps)
        x1, mapping = sort(x)
        inverse_mapping = invert(mapping)
        # print(x, x1, eps/n)

        S = set()
        if x1[0] >= eps/len(self):
            return S
        else:
            S.add(0)

        for i in range(1, len(self)):
            if x1[i] >= 0 and x1[i] - x1[i - 1] >= eps/n:
                return S
            else:
                S.add(inverse_mapping[i])

        return S


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
    def __init__(self, n: int, U_neighbors):
        super().__init__(n)
        self.U = list(range(n))
        self.V = list(range(n))
        self.U_neighbors = U_neighbors

    def function_value(self, T: set):
        """
        Get f(T) = w(N(T)), the total weight of neighborhoods of T
        :param T: subset of V
        :return: f(T) = w(N(T))
        """
        N = set()
        for t in T:
            N = N.union(set(self.U_neighbors[t]))

        return len(N)


class RestrictedSubmodularFunction(SubmodularFunction):
    def __init__(self, f: SubmodularFunction, r, S=None, T=None):
        super().__init__()
        self.n = len(T) - len(S)
        if not S.issubset(T):
            raise ValueError('S should be a subset of T')
        self.f0 = f
        self.S = S
        self.T = T
        self.mapping = self.get_mapping()
        self.inverse_mapping = self.get_inverse_mapping()
        self.r = r

    def get_mapping(self):
        count = 0
        mapping = {}
        for e in self.T.difference(self.S):
            mapping[e] = count
            count = count + 1

        # print(mapping)
        return mapping

    def get_inverse_mapping(self):
        return {v: k for k, v in self.mapping.items()}

    def function_value(self, W: set):
        # logging.warning(str(W) + str(self.T) + str(self.S))

        W1 = map_set(W, self.inverse_mapping)
        return self.f0.function_value(W1.union(self.S)) - self.f0.function_value(self.S) \
            - sum([self.r[e] for e in W1])


class RestrictedSubmodularPolytope(SubmodularPolytope):
    def __init__(self, f: RestrictedSubmodularFunction):
        self.f = f
        self.n = len(f.T) - len(f.S)

    def maximal_minimizer(self, eps):
        x = self.minimum_norm_point_using_afw(eps)
        x1, mapping = sort(x)
        inverse_mapping = invert(mapping)
        print('Max min', x, x1, self.f.S, self.f.T)

        S = set()
        if x1[0] >= eps/self.n:
            inverse_mapping_for_coordinates = self.f.inverse_mapping
            S = map_set(S, inverse_mapping_for_coordinates).union(self.f.S)
            return S
        else:
            S.add(0)

        for i in range(1, self.n):
            if x1[i] >= 0 and x1[i] - x1[i - 1] >= eps/self.n:
                for j in range(i, self.n):
                    if abs(x1[j] - x1[i]) <= 0.0001:
                        S.add(inverse_mapping[j])
                    else:
                        break
                print('S here', S)
                inverse_mapping_for_coordinates = self.f.inverse_mapping
                S = map_set(S, inverse_mapping_for_coordinates).union(self.f.S)
                return S
            else:
                S.add(inverse_mapping[i])

        inverse_mapping_for_coordinates = self.f.inverse_mapping
        S = map_set(S, inverse_mapping_for_coordinates).union(self.f.S)

        return S


def create_bipartite_graph(n: int, p: float):
    U_neighbors = []
    for u in range(n):
        neighbors = []
        for v in range(n):
            q = random.random()
            if q < p:
                neighbors.append(v)

        U_neighbors.append(neighbors)

    return U_neighbors


def integer_rounding(x: List[float]):
    t = len(x)
    rounded_point = [0.0] * t
    for i in range(t):
        min_distance = abs(round(x[i]) - x[i])
        rounded_number = round(x[i])
        for j in range(2, t + 1):
            if abs((round(x[i] * j) - x[i] * j)/j) < min_distance:
                min_distance = abs((round(x[i] * j) - x[i] * j)/j)
                rounded_number = round(x[i] * j)/j

        rounded_point[i] = rounded_number

    return rounded_point


def da_projection(y, P: SubmodularPolytope, initial_tight_sets=None):
    n = len(P)
    eps = 1/(n*n)

    def da_step(S, T):
        """
        Performs DA(S, T)
        :param S: lower set
        :param T: upper set, assumed to be a superset of S
        :return: projection of y on x over the restricted polytope, as described in
        Nagano-Aihara's DA
        """

        s = sum([y[e] for e in T.difference(S)])
        alpha = (P.f.function_value(T) - P.f.function_value(S) - s)/(len(T) - len(S))
        print('sets, input, alpha', S, T, y, alpha)

        # Form restricted coordinate set:
        mapping = {}
        count = 0
        for e in T.difference(S):
            mapping[e] = count
            count = count + 1

        x_alpha = [y[e] + alpha for e in range(n)]
        # f_alpha = DifferenceSubmodularFunction(f, x_alpha)
        f_restricted = RestrictedSubmodularFunction(f, x_alpha, S, T)
        P_restricted = RestrictedSubmodularPolytope(f_restricted)
        R = P_restricted.maximal_minimizer(eps)
        print(R)

        print(S, R, T, x_alpha)

        if R == T:
            return x_alpha
        else:
            x1 = da_step(S, R)
            x2 = da_step(R, T)
            for e in range(n):
                if e in R.difference(S):
                    x_alpha[e] = x1[e]
                if e in T.difference(R):
                    x_alpha[e] = x2[e]

            return x_alpha

    if initial_tight_sets is None:
        initial_tight_sets = [set(), set(range(n))]

    x = da_step(set(), set(range(n)))
    return x


random.seed(0)

# n = 2
# f = PermutahedronSubmodularFunction(n)
# P = Permutahedron(f)
# print(P.minimum_norm_point_using_afw(eps=0.01))
# U_neighbors = create_bipartite_graph(5, 0.6)
# print(U_neighbors)
#
# f = BiparititeGraphSubmodularFunction(n, U_neighbors)
#
# P = SubmodularPolytope(f)
#
# T = [set(), {4, 1}, {0, 1, 2, 3, 4}]
# print(P.linear_optimization_tight_sets([0.1, 0.4, 1.0, 0.8, 0.8], T))

n = 4
f = PermutahedronSubmodularFunction(n)
P = Permutahedron(f)
y = np.array([1.0, 3.0, 2.0, -4.0]) #, 4.0, 5.0, 1.0, 0.0, 0.0, -6.0])
# y = np.zeros(n)
x = da_projection(y, P)


from continuous_methods import isotonic_projection
g = {0: 0.0}
for i in range(n):
    g[i + 1] = g[i] + (n - i)

x1 = isotonic_projection(y, g)
print('Results')
print(x, x - y)
print(x1, x1 - y)
print(determine_tight_sets(y, x1, g))
print(np.linalg.norm(x1 - y), np.linalg.norm(x - y))

