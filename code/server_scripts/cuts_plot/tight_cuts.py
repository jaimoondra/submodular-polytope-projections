import logging
logging.basicConfig(level=logging.DEBUG)

import matplotlib
import pandas as pd
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools


# Constants:
n = 100  # size of ground set
base_decimal_accuracy = 5  # All inputs and outputs will have this accuracy
minimum_decimal_difference = math.pow(10, -base_decimal_accuracy)  # for float accuracy
high_decimal_accuracy = 12  # Used only by inner computations in algorithms
dist_square = 50  # todo: add description
std_dev_point = 5  # Standard deviation for selection on random point
seeds = [1234, 90, 0, 6454, 6, 444444, 39256, 7527, 50604, 24743, 47208, 28212, 19019, 41225, 23406,
         52847, 62727, 3034, 55949, 13206, 8086, 55396, 21709, 10223, 41131, 45982, 51335, 19036,
         9056, 17681, 15141, 6306, 63724, 42770, 35394, 44056, 22564, 50203, 13494, 2617, 62882,
         35918, 2597, 43039, 7228, 35110, 63328, 35294, 21347, 69, 55129, 64711, 24826, 25899,
         13623, 64414, 18845, 51362, 15405, 39271, 29175, 31418, 3071, 9840, 49312, 63306, 48069,
         48216, 59896, 52064, 7533, 9390, 36907, 25146, 7840, 42243, 35634, 50032, 12157, 47424,
         39071, 9496, 30727, 11739, 60247, 33845, 25754, 45533, 27374, 29006, 3133, 8072, 6823,
         55874, 54767, 29723, 50573, 19110, 40861, 17731, 20386, 54415, 11486, 63471, 26744, 3881]
# Multiple seeds to ensure both randomness and repeatability




def pool(values, weights, l, r, ):
    new_point = sum(map(lambda x: values[x] * weights[x], range(l, r + 1))) / sum(weights[l: r + 1])
    values[l] = new_point
    weights[l] = sum(weights[l: r + 1])

    return values[:l + 1], weights[:l + 1]


def poolAdjacentViolators(input):
    """
    Main function to solve the pool adjacent violator algorithm
    on the given array of data.
    This is a O(n) implementation. Trick is that while regersssing
    if we see a violation, we average the numbers and instead of
    storing them as two numbers, we store the number once and store
    a corresponding weight. This way, for new numbers we don't have
    to go back for each n, but only one place behind and update the
    corresponding weights.
    """
    weights = []
    output = []

    index = 0
    while index < len(input):
        temp = index

        # Find monotonicity violating sequence, if any.
        # Difference of temp-beg would be our violating range.
        while temp < len(input) - 1 and input[temp] > input[temp + 1]:
            # Append this number to the final output and set its weight to be 1.
            temp += 1

        if temp == index:
            output_beg = len(output) - 1
            output_end = output_beg + 1
            output.append(input[index])
            weights.append(1)
            index += 1
        else:
            # Pool the violating sequence, if after violating monotonicity
            # is broken, we need to fix the output array.
            output_beg = len(output)
            output_end = output_beg + temp - index
            output.extend(input[index: temp + 1])
            weights.extend([1] * (temp - index + 1))
            index = temp + 1

        # Fix the output to be in the increasing order.
        while output_beg >= 0 and output[output_beg] > output[output_beg + 1]:
            output, weights = pool(output, weights, output_beg, output_end)
            diff = (output_end - output_beg)
            output_beg -= 1
            output_end -= diff

    return np.array(
        list(itertools.chain(*map(lambda i: [output[i]] * weights[i], range(len(weights))))))


def isotonic_projection(y, g):
    """
    Main function to compute a projection over cardinality based B(f).
    This functions solved the dual problem in the dual space using the
    PAV algorithm and then maps it back to the primal. The inputs are
    the point y we are trying to project and a dictionary g with the
    submodular function. g is of the form {0:0, 1:f(1),..., n:f(n)}
    """

    n = len(g) - 1
    pi = np.argsort(-y)
    C = np.array([g[i] - g[i - 1] for i in range(1, n + 1)])
    C_ = {}
    for i, j in enumerate(pi):
        C_[j] = C[i]

    C_ = np.array([C_[i] for i in sorted(C_.keys())])

    error = C_ - y
    error_sorted = error[pi]
    z = poolAdjacentViolators(error_sorted)

    z_ = {}
    for i, j in enumerate(pi):
        z_[j] = z[i]

    return np.array([z_[i] for i in sorted(z_.keys())]) + y



def round_list(x, accuracy: int = base_decimal_accuracy):
    """
    Round a list of numbers to appropriate accuracy
    :param x: List of numbers
    :param accuracy: Number of decimal digits to round float numbers to
    :return: List of rounded numbers
    """
    x = [round(i, accuracy) for i in x]
    return x


def sort(x, reverse=False):
    """
    :param x: List of numbers
    :param reverse: Sorts in decreasing order if set to True
    :return: Sorted list and the corresponding mapping (permutation)
    """
    enum = sorted(enumerate(x), key=lambda z: z[1], reverse=reverse)
    y = [enum[j][1] for j in range(len(enum))]
    mapping = {enum[j][0]: j for j in range(len(enum))}

    return y, mapping


def invert(mapping):
    """
    Invert a (bijective) mapping {0, ..., n - 1} -> {0, ..., n - 1}
    :param mapping: Original mapping
    :return: Inverse of the original mapping
    """
    return {mapping[i]: i for i in range(len(mapping))}


def map_set(S, mapping):
    """
    Determines the range of S under mapping
    :param S: set of integers
    :param mapping: mapping
    :return: range of S under mapping as a set
    """
    return set({mapping[i] for i in S})


def permute(x, mapping):
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


def plot_iterates(iterates, y, color1, color2):
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


def plot_tight_sets(tight_sets, color1):
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

class SubmodularFunction:
    def __init__(self, n: int = 0):
        """
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
    def __init__(self, g, r, n: int = 0):
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
    def is_cardianality_submodular(g):
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
    def __init__(self, g, n):
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

    def linear_optimization_over_base(self, c):
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
    def check_chain(T):
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

    def linear_optimization_tight_sets(self, c, T):
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
        # print('T = ', T)
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
                D_up_inverse = map_set(set(range(i + 1)), inverse)
                D_up_inverse = set([a + l for a in list(range(i + 1))])

                D_down_inverse = map_set(set(range(i)), inverse)
                D_down_inverse = set([a + l for a in list(range(i))])

                S_up = T[j - 1].union(map_set(D_up_inverse, inverse_permutation))
                S_down = T[j - 1].union(map_set(D_down_inverse, inverse_permutation))

                y.append(self.f.function_value(S_up) - self.f.function_value(S_down))
                opt = opt + d[i] * y[-1]
            y = permute(y, inverse)
            x = x + y
            l = l + m

        z = permute(x, invert(permutation))
        return opt, z

    def affine_minimizer(self, S):
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

        def nonnegative_coordinates(y):
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

    def is_feasible(self, x):
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

    def is_feasible_in_base(self, x):
        """
        Checks if x in is B(f)
        :param x: point in space
        :return: True if x is in B(f), False otherwise
        """
        return True if sum(x) == self.f.g[-1] and self.is_feasible(x) else False

    @staticmethod
    def prefix_sum(x):
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


def close_points_learning(n: int, m: int = 40, seed=None, g=None, epsilon=0.1):
    """
    Uses algorithm described after Lemma 7 to determine save in running time
    :param n: Size of the ground set
    :param m: Number of points in the neighborhood
    :param seed: seed for random
    """
    np.random.seed(seed)

    def generate_close_random_point(y0):
        error = random_error(n=n, r=epsilon, seed=None, decial_accuracy=6)
        y = np.array([(y0[i] + error[i]) for i in range(n)])
        return y

    def seen_tight_sets_count(tight_sets0, tight_sets1):
        """
        Infers tights sets for tilde(y) from tight sets for y
        :param tight_sets1: tight_sets of point y, along with their c[j + 1] - c[j] values,
        sorted by their c[j + 1] - c[j] values
        :return: inferred tight sets for tilde(y), list
        """
        inferred_tight_sets = set()

        tight_sets_list0 = {tight_sets0[t][1] for t in range(len(tight_sets0))}
        tight_sets_list1 = {tight_sets1[t][1] for t in range(len(tight_sets1))}

        return tight_sets_list0.intersection(tight_sets_list1)

    def infer_tight_sets_from_close_point(d, tight_sets1):
        """
        Infers tights sets for tilde(y) from tight sets for y
        :param d: distance d(y, tilde(y))
        :param tight_sets1: tight_sets of point y, along with their c[j + 1] - c[j] values,
        sorted by their c[j + 1] - c[j] values
        :return: inferred tight sets for tilde(y), list
        """
        inferred_tight_sets = set()

        for t in range(len(tight_sets1)):
            a = tight_sets1[t]
            if 4 * d < a[0]:
                inferred_tight_sets.add(a[1])
            else:
                break

        return inferred_tight_sets

    def check_tight_sets(tight_sets_inferred, tight_sets_for_first_point):
        """
        Checks if inferred tight sets are a subset of actual tight sets
        :param tight_sets_inferred:
        """
        y1 = points[j]
        x1 = projections[j]

        if len(tight_sets_for_first_point) == 0:
            return

        tight_sets_for_first_point = set(list(tight_sets_for_first_point[:, 1]))
        sets_to_remove = []
        for s in tight_sets_inferred:
            if s not in tight_sets_for_first_point:
                l = len(s)
                x_s = 0.0
                x1_s = 0.0
                for k in s:
                    x_s = x_s + x[k]
                    x1_s = x1_s + x1[k]
                if abs(g[l] - x_s) < 0.1:
                    sets_to_remove.append(s)
                else:
                    logging.info('Danger, rogue set identified')
                    logging.debug('Tight set: ' + str(s))
                    logging.debug('Submodular function: ' + str(g))
                    logging.debug('Submodular function value: ' + str(g[l]))
                    logging.debug('First point: ' + str(y1) + ', ' + str(x1) + ', ' + str(x1_s))
                    logging.debug('New point: ' + str(y) + ', ' + str(x) + ', ' + str(x_s))
                    logging.debug('Distance: ' + str(d))

                    raise ValueError('Danger, rogue tight set identified.')

        for s in sets_to_remove:
            tight_sets_inferred.remove(s)

        return

    # Submodular function f(S) := g(|S|). Generate randomly if not given.
    if g is None:
        g = generate_concave_function(n)
        g = [0] + g
    g = {i: g[i] for i in range(n + 1)}

    # Table for storing points and tight_sets:
    points, projections, tight_sets_list = [], [], []

    # Central random point
    logging.info('Point #' + str(0))

    t = np.random.randint(1, n)
    c = np.random.randint(1, t/2 + 2)
    y0 = np.array(random_point(n=n, mean=t, r=c, nonnegative=True, seed=seed))
    points.append(y0)

    # Calculate projection
    x0 = np.array(isotonic_projection(y0, g))
    projections.append(x0)
    tight_sets_list.append(determine_tight_sets(y0, x0, g))

    learned_tight_sets = [0]
    seen_tight_sets = [0]
    total_tight_sets = [len(tight_sets_list[0])]

    for i in range(1, m + 1):
        logging.info('Point #' + str(i))

        y = generate_close_random_point(y0)
        x = isotonic_projection(y, g)
        points.append(y)
        projections.append(x)
        tight_sets = determine_tight_sets(y, x, g)
        tight_sets_list.append(tight_sets)

        inferred_tight_sets = set()
        seen_tight_sets_for_this_point = set()
        for j in range(i):
            seen_tight_sets_for_this_point = seen_tight_sets_for_this_point.union(
                seen_tight_sets_count(tight_sets, tight_sets_list[j]))
            d = np.linalg.norm(y - points[j])
            inferred_tight_sets =\
                inferred_tight_sets.union(infer_tight_sets_from_close_point(d, tight_sets_list[j]))

        check_tight_sets(inferred_tight_sets, tight_sets)

        learned_tight_sets.append(learned_tight_sets[-1] + len(inferred_tight_sets))
        total_tight_sets.append(total_tight_sets[-1] + len(tight_sets))
        seen_tight_sets.append(seen_tight_sets[-1] + len(seen_tight_sets_for_this_point))

    fractional_learned_tight_sets = np.array([learned_tight_sets[i] / total_tight_sets[i]
                                                 for i in range(1, m + 1)])
    fractional_seen_tight_sets = np.array([seen_tight_sets[i] / total_tight_sets[i]
                                              for i in range(1, m + 1)])
    return learned_tight_sets, seen_tight_sets, total_tight_sets, fractional_learned_tight_sets, \
           fractional_seen_tight_sets


n = 100
m = 500
f = PermutahedronSubmodularFunction(n)
P = CardinalityPolytope(f)
outer = 500

# matplotlib.rcParams.update({'font.size': 15})
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

f = PermutahedronSubmodularFunction(n)
P = CardinalityPolytope(f)
g = P.f.g

average = np.zeros(m)
average_seen = np.zeros(m)

df_learned = pd.DataFrame()
df_seen = pd.DataFrame()

for t in range(outer):
    logging.info('Iteration ' + str(t))

    a, b, c, fractional_learned_tight_sets, fractional_seen_tight_sets =\
        close_points_learning(n, m, seed=t, g=g, epsilon=1/n)

    df_learned['Run ' + str(t)] = fractional_learned_tight_sets
    df_seen['Run ' + str(t)] = fractional_seen_tight_sets
    # print(a, b, fractional_learned_tight_sets)
    # average = (t/(t + 1)) * average + (1/(t + 1)) * fractional_learned_tight_sets
    # average_seen = (t / (t + 1)) * average_seen + (1 / (t + 1)) * fractional_seen_tight_sets
    # cyan, = plt.plot(fractional_learned_tight_sets, color='cyan', linewidth=0.3)
    # orange, = plt.plot(fractional_seen_tight_sets, color='orange', linewidth=0.3)

# blue, = plt.plot(average)
# red, = plt.plot(average_seen)


df_learned.to_csv(r'learned_n_100_noise_n.csv', index=False)
df_learned.to_csv(r'seen_n_100_noise_n.csv', index=False)

# quantile10_learned = df_learned.quantile(0.15, axis=1)
# quantile10_seen = df_seen.quantile(0.15, axis=1)
# quantile90_learned = df_learned.quantile(0.85, axis=1)
# quantile90_seen = df_seen.quantile(0.85, axis=1)
# average_learned = df_learned.mean(axis=1)
# average_seen = df_seen.mean(axis=1)
#
#
# print(quantile90_seen, quantile90_learned, quantile10_seen, quantile10_learned)
#
#
# plt.title(r'Tight sets from close points for $n = 100$, noise $\frac{1}{n}$')
# plt.xlabel(r'Iterations')
# plt.ylabel(r'Fraction of tight sets for close points')
# plt.savefig('tight_cuts_seen_using_gradient_for_close_points_n_100_error_n.png', dpi=300)
# plt.fill_between(range(m), quantile10_seen, quantile90_seen, color='green',
#                  alpha=0.1)
# plt.fill_between(range(m), quantile10_learned, quantile90_learned, color='blue',
#                  alpha=0.1)
# learned, = plt.plot(average_learned, color='blue')
# seen, = plt.plot(average_seen, color='green')
# learned.legend(r'Fraction of tight sets common with a previous point.')
# seen.legend(r'Fraction of tight sets recovered from a previous point.')
#
# plt.show()


# print(df_learned)
# print(df_seen)
