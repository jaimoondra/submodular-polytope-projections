import logging

logging.basicConfig(level=logging.WARNING)

import matplotlib.pyplot as plt
import itertools
import math
import numpy as np
import random
import time
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from gurobipy import *
import numpy.linalg as npl
from time import process_time
import datetime
import sys


# Constants:
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


def active_set_oracle(vertices, x):
    try:
        m = Model("opt")
        vertices = list(vertices)
        n = len(vertices[0])

        # define variables
        lam = {}
        for i in range(len(vertices)):
            lam[i] = m.addVar(lb=0, name='lam_{}'.format(i),obj = 1)

        # feasibility constraints
        for i in range(n):
            m.addConstr(x[i],'=', sum([lam[j] * vertices[j][i] for j in range(len(vertices))]))

        # convex hull constraint
        m.addConstr(quicksum([lam[i] for i in lam.keys()]), '=',1)
        m.update()

        # optimize
        m.setParam( 'OutputFlag', False )
        # m.write('exact.lp')
        m.optimize()
        v = {}

        for i in lam:
            if np.round(lam[i].x,5) > 0:
                v[tuple(vertices[i])] = lam[i].x

        return True

    except AttributeError:
        return False


# line-search using golden-section rule
def segment_search(f, grad_f, x, y, tol=1e-6, stepsize=True):
    '''
    Minimizes f over [x, y], i.e., f(x+gamma*(y-x)) as a function of scalar gamma in [0,1]
    '''

    # restrict segment of search to [x, y]
    d = (y - x).copy()
    left, right = x.copy(), y.copy()

    # if the minimum is at an endpoint
    if np.dot(d, grad_f(x)) * np.dot(d, grad_f(y)) >= 0:
        if f(y) <= f(x):
            return y, 1
        else:
            return x, 0

    # apply golden-section method to segment
    gold = (1 + np.sqrt(5)) / 2
    improv = np.inf
    while improv > tol:
        old_left, old_right = left, right
        new = left + (right - left) / (1 + gold)
        probe = new + (right - new) / 2
        if f(probe) <= f(new):
            left, right = new, right
        else:
            left, right = left, probe
        improv = np.linalg.norm(f(right) - f(old_right)) + np.linalg.norm(f(left) - f(old_left))
    x_min = (left + right) / 2

    # compute step size gamma
    gamma = 0
    if stepsize == True:
        for i in range(len(d)):
            if d[i] != 0:
                gamma = (x_min[i] - x[i]) / d[i]
                break

    return x_min, gamma


def line_search(x, d, gamma_max, func):
    def fun(gamma):
        ls = x + gamma * d
        return func(ls)

    res = minimize_scalar(fun, bounds=(0, gamma_max), method='bounded')

    gamma = res.x
    ls = x + gamma * d
    return ls, gamma


def AFW(x, S, lmo, epsilon, func, grad_f, f_tol, time_tol):
    n = len(x)

    # Fucntion to compute away vertex
    def away_step(grad, S):
        costs = {}

        for k, v in S.items():
            cost = np.dot(k, grad)
            costs[cost] = [k, v]
        vertex, alpha = costs[max(costs.keys())]
        return vertex, alpha

    # Function to update the active set
    def update_S(S, gamma, Away, vertex, x):
        S = S.copy()
        vertex = tuple(vertex)

        if not Away:
            if vertex not in S.keys():
                S[vertex] = gamma
            else:
                S[vertex] *= (1 - gamma)
                S[vertex] += gamma

            for k in S.keys():
                if k != vertex:
                    S[k] *= (1 - gamma)
        else:
            for k in S.keys():
                if k != vertex:
                    S[k] *= (1 + gamma)
                else:
                    S[k] *= (1 + gamma)
                    S[k] -= gamma

        # Drop any vertices whose coefficients fall below 10^{-8}
        T = {k: v for k, v in S.items() if np.round(v, 10) > 0}

        # Update the point x

        x = np.zeros(n)
        for k in T:
            x = x + T[k] * np.array(k)
        t = sum(list(T.values()))
        x = x/t
        # Update coefficients in set T
        T = {k: T[k]/t for k in T}

        return T, x

    # record primal gap, function value, and time every iteration
    now = datetime.datetime.now()
    primal_gap = []
    function_value = [func(x)]
    time = [0]
    f_improv = np.inf
    # initialize starting point and active set
    t = 0
    while f_improv > f_tol and time[-1] < time_tol:
        start = process_time()
        # compute gradient
        grad = grad_f(x)
        # solve linear subproblem and compute FW direction
        v = lmo(-grad)
        d_FW = v - x
        # If primal gap is small enough - terminate
        # logging.warning('Primal gap:' + str(np.dot(-grad, d_FW)))
        if np.dot(-grad, d_FW) <= epsilon:
            end = process_time()  # Record end time
            time.append(time[t] + end - start)
            t = t + 1
            break
        else:
            # update convergence data
            primal_gap.append(np.dot(-grad, d_FW))
            logging.info(str(np.dot(-grad, d_FW)))
            logging.debug(str(v))

        # Compute away vertex and direction
        a, alpha_a = away_step(grad, S)
        d_A = x - a
        # Check if FW gap is greater than away gap
        if np.dot(-grad, d_FW) >= np.dot(-grad, d_A):
            # choose FW direction
            d = d_FW
            vertex = v
            gamma_max = 1
            Away = False
        else:
            # choose Away direction
            d = d_A
            vertex = a
            gamma_max = alpha_a / (1 - alpha_a)
            Away = True
        # Update next iterate by doing a feasible line-search

        # y = x + gamma_max * d
        # # restrict segment of search to [x, y]
        # d = (y - x).copy()
        # left, right = x.copy(), y.copy()

        # if the minimum is at an endpoint
        if np.dot(d, grad_f(x + gamma_max * d)) <= 0:
            gamma = gamma_max
            x = x + gamma * d
        else:
            x, gamma = line_search(x, d, gamma_max, func)
        # x, gamma = segment_search(func, grad_f, x, x + gamma_max *d)

        # update active set
        S, x = update_S(S, gamma, Away, vertex, x)

        end = process_time()                        # Record end time
        time.append(time[t] + end - start)          # Update time taken in this iteration
        f_improv = function_value[-1] - func(x)     # Update improvement in function value
        function_value.append(func(x))              # Record current function value
        # logging.warning('Function value:' + str(func(x)))
        t += 1

    return x, function_value, time[1:], t, primal_gap, S


def adaptive_AFW_cardinality_polytope(x, S, P, epsilon, func, grad_f, f_tol, time_tol,
                                      initial_tight_sets, y):
    """
    Adaptive AFW
    :param x: initial vertex
    :param S: initial active set
    :param P: Submodular Polytope
    :param epsilon: AFW-gap parameter
    :param func: function to minimize (Euclidean distance in our case)
    :param grad_f: gradient of func
    :param f_tol:
    :param time_tol: Max allowed time
    :param initial_tight_sets: Known tight sets
    :param y: point to be projected
    :return:
    """
    # Fucntion to compute away vertex
    n = len(x)
    T = set(S.keys())
    def away_step(grad, S):
        costs = {}

        for k, v in S.items():
            cost = np.dot(k, grad)
            costs[cost] = [k, v]
        vertex, alpha = costs[max(costs.keys())]
        return vertex, alpha

    def update_S(S, gamma, Away, vertex, x):
        S = S.copy()
        vertex = tuple(vertex)

        if not Away:
            if vertex not in S.keys():
                S[vertex] = gamma
            else:
                S[vertex] *= (1 - gamma)
                S[vertex] += gamma

            for k in S.keys():
                if k != vertex:
                    S[k] *= (1 - gamma)
        else:
            for k in S.keys():
                if k != vertex:
                    S[k] *= (1 + gamma)
                else:
                    S[k] *= (1 + gamma)
                    S[k] -= gamma

        # Drop any vertices whose coefficients fall below 10^{-8}
        U = {k: v for k, v in S.items() if np.round(v, 10) > 0}

        # Update the point x
        x = np.zeros(n)
        for k in U:
            x = x + U[k] * np.array(k)
        t = sum(list(U.values()))
        x = x/t
        # Update coefficients in set T
        U = {k: U[k]/t for k in U}

        return U, x

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
            if 2 * d < a[0]:
                inferred_tight_sets.add(a[1])
            else:
                break

        return inferred_tight_sets

    # record primal gap, function value, and time every iteration
    now = datetime.datetime.now()
    primal_gap = []
    function_value = [func(x)]
    time = [0]
    f_improv = np.inf
    t = 0
    inferred_tight_sets = initial_tight_sets.union({frozenset()})

    while f_improv > f_tol and time[-1] < time_tol:
        # solve linear subproblem and compute FW direction
        def lmo(c, inferred_tight_sets):
            inferred_tight_sets_list = list(inferred_tight_sets)
            inferred_tight_sets_list.sort(key=len)
            _, v = P.linear_optimization_tight_sets(c, inferred_tight_sets_list)
            return v

        start = process_time()              # Start time
        grad = grad_f(x)                    # Compute gradient

        v = lmo(-grad, list(inferred_tight_sets))
        d_FW = v - np.array(x)

        gap = np.dot(-grad, d_FW)

        # if gap <= epsilon:  # If primal gap is small enough - terminate
        #     end = process_time()  # Record end time
        #     time.append(time[t] + end - start)
        #     t = t + 1
        #     break

        if gap < 0:
            gap = 0

        d = 2 * math.sqrt(gap)
        tight_sets = determine_tight_sets(y, x)
        new_inferred_tight_sets = infer_tight_sets_from_close_point(d, tight_sets)
        if not new_inferred_tight_sets.issubset(inferred_tight_sets):
            inferred_tight_sets = inferred_tight_sets.union(new_inferred_tight_sets)
            x = lmo(-grad, list(inferred_tight_sets))
            T = T.union(S.keys())
            # print(T)
            S = {tuple(x): 1}
            try:
                x = cut_rounding(P, list(inferred_tight_sets), y, T)
                end = process_time()
                time.append(time[t] + end - start)
                f_improv = function_value[-1] - func(x)
                function_value.append(func(x))
                t += 1
                logging.warning('Rounded successfully!')
                return x, function_value, time, t, primal_gap, S, inferred_tight_sets
            except ValueError:
                if gap <= epsilon:  # If primal gap is small enough - terminate
                    break
                pass

        else:
            if gap <= epsilon:                      # If primal gap is small enough - terminate
                end = process_time()  # Record end time
                time.append(time[t] + end - start)
                t = t + 1
                break
            else:
                primal_gap.append(gap)              # update convergence data
                logging.info(str(gap))
                logging.debug(str(v))

            # Compute away vertex and direction
            a, alpha_a = away_step(grad, S)
            d_A = np.array(x) - np.array(a)

            # Check if FW gap is greater than away gap
            if np.dot(-grad, d_FW) >= np.dot(-grad, d_A):
                d = d_FW                            # choose FW direction
                vertex = v
                gamma_max = 1
                Away = False
            else:
                d = d_A                             # choose Away direction
                vertex = a
                gamma_max = alpha_a / (1 - alpha_a)
                Away = True

            # Update next iterate by doing a feasible line-search
            # x, gamma = line_search(x, d, gamma_max, func)

            # if the minimum is at an endpoint
            if np.dot(d, grad_f(x + gamma_max * d)) <= 0:
                gamma = gamma_max
                x = x + gamma * d
            else:
                x, gamma = line_search(x, d, gamma_max, func)

            # update active set
            S, x = update_S(S, gamma, Away, vertex, x)
        end = process_time()
        time.append(time[t - 1] + end - start)
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        t += 1

    return x, function_value, time[1:], t, primal_gap, S, inferred_tight_sets

def convex_hull_correction1(S, func):
    M = np.array([np.array(i) for i in S])

    def fun(theta):
        return func(np.dot(M.T, theta))

    cons = ({'type': 'eq', 'fun': lambda theta: sum(theta) - 1})  # sum of theta = 1
    bnds = tuple([(0, 1) for _ in M])
    x0 = tuple([1 / len(M) for _ in M])

    res = minimize(fun, x0, bounds=bnds, constraints=cons)

    final_S = {tuple(M[i]): res.x[i] for i in range(len(M)) if np.round(res.x[i], 5) > 0}

    return np.dot(M.T, res.x), final_S


def convex_hull_correction2(S, q):
    M = np.array([np.array(i) for i in S])

    opt, theta = proj_oracle(M, q)

    final_S = {tuple(M[i]): theta[i] for i in range(len(M)) if np.round(theta[i], 5) > 0}

    return opt, final_S


def proj_oracle(vertices, y):
    m = Model("opt")
    n = len(vertices[0])

    # define variables
    lam = {}
    for i in range(len(vertices)):
        lam[i] = m.addVar(lb=0, name='lam_{}'.format(i))

    x = []
    for i in range(n):
        x.append(m.addVar(lb=-GRB.INFINITY, name='x_{}'.format(i)))
    x = np.array(x)
    m.update()

    objExp = 0.5 * np.dot(x - y, x - y)
    m.setObjective(objExp, GRB.MINIMIZE)
    m.update()

    # feasibility constraints
    for i in range(n):
        m.addConstr(x[i], '=', sum([lam[j] * vertices[j][i] for j in range(len(vertices))]))

    # convex hull constraint
    m.addConstr(quicksum([lam[i] for i in lam.keys()]), '=', 1)
    m.update()

    # optimize
    m.setParam('OutputFlag', False)
    # m.write('exact.lp')
    m.optimize()
    return np.array([i.x for i in x]), np.array([lam[i].x for i in lam])


def generate_loss_functions_for_permutahedron(a, b, seed, n: int, T: int):
    """
    Generates loss functions
    :param a: number of permutations
    :param b: max swap distance between any two permutations
    :param seed: seed for random
    :param n: size of ground set
    :param T: time (number of iterations in the online problem)
    :return: loss function vectors
    """
    np.random.seed(seed)
    random.seed(seed)

    def generate_close_permutation():
        sigma1 = np.array([sigma0[j] for j in range(n)])
        for j in range(b):
            c, d = np.random.randint(0, n), np.random.randint(0, n)
            sigma1[c], sigma1[d] = sigma1[d], sigma1[c]

        return sigma1

    sigma0 = np.array(generate_random_permutation(n))
    permutations = [sigma0]
    for i in range(1, a):
        sigma = generate_close_permutation()
        permutations.append(sigma)

    loss_vectors_list = []
    for i in range(T):
        x = [random.random() for i in range(n)]  # Each entry is random number between 0 and 1
        s = sum(x)
        x = round_list([(x[i] * n) / s for i in range(n)], 3)  # Normalize so the sum is always n
        x.sort()

        loss = [0.0] * n
        t = np.random.randint(0, a)
        sigma = permutations[t]
        for j in range(n):
            loss[j] = x[sigma[j] - 1]

        loss_vectors_list.append(np.array(loss))

    return loss_vectors_list


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


class SubmodularFunction:
    """
    Class to represent submodular functions
    """
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
    """
    Class to represent a submodular function of the form f1(W) = f(W) - r(W), where r is a vector
    and f is given
    """
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
    """
    Class for cardinality-based submodular functions
    """
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
    """
    Class to represent submodular polytopes
    """
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
        # c = round_list(c, 10)
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


def tight_set_rounded_solution(C: CardinalityPolytope, H, y):
    """
    T is assumed to contain the empty set and the ground set, and is assumed to be a chain.
    :param C:
    :param T:
    :return:
    """
    n = len(C)
    x = np.zeros(n)

    for j in range(1, len(H)):
        F = H[j].difference(H[j - 1])
        y_F = sum([y[k] for k in F])
        alpha = (C.f.function_value(H[j]) - C.f.function_value(H[j - 1]) - y_F)/len(F)
        for i in F:
            x[i] = alpha + y[i]

    return x


def cut_rounding(C: CardinalityPolytope, H, y, S):
    """
    Function for tight-set based round algorithm
    :param C: Cardinality-based polytope
    :param H:
    :param y:
    :param S:
    :return:
    """
    H.sort(key=len)

    x = tight_set_rounded_solution(C, H, y)
    is_convex_combination = active_set_oracle(S, x)

    if is_convex_combination:
        return x
    else:
        raise ValueError('Cannot round, x not feasible.')


def sample_vertex_from_convex_combination(S):
    vertices = list(S.keys())
    s = len(S)
    distribution = np.array(list(S.values()))
    sum_distribution = sum(distribution)
    distribution = (1 / sum_distribution) * distribution
    v = np.random.choice(s, p=distribution)
    return np.array(vertices[v])


def projection_on_permutahedron_using_afw_euclidean(n, y, epsilon, lmo, S=None, x=None):
    """
    :param n: |E|
    :param y: Point to project
    :param epsilon: Error
    :param S: Ative set dict
    :param x: Intial iterate
    :return:
    """
    y = np.array(y)
    h = lambda x: 0.5 * np.dot(x - y, x - y)
    grad_h = lambda x: np.power(x - y, 1)
    h_tol, time_tol = -1, np.inf

    if x is None:
        w = generate_random_permutation(n)
        x = w
        S = {tuple(w): 1}

    A = AFW(np.array(x), S, lmo, epsilon, h, grad_h, h_tol, time_tol)
    # print(A)
    return A


def vanilla_mirror_descent(n, epsilon, P, T, loss_vectors_list, x_0, eta):
    total_time = 0.0
    fw_iterations = []
    step_regret = []
    time_steps = []
    opt, alg = 0.0, 0.0

    def lmo(c):
        _, v = P.linear_optimization_over_base(c)
        return tuple(v)

    S = {x_0: 1}
    x = x_0

    for t in range(T):
        logging.warning('Iteration ' + str(t))
        # sigma = sample_vertex_from_convex_combination(S)
        sigma = x

        l = loss_vectors_list[t]
        loss = np.dot(l, sigma)
        alg = loss

        sol, x_star = P.linear_optimization_over_base(list(-l))

        opt = np.dot(l, x_star)
        step_regret.append(alg - opt)

        y = x - eta * l

        x, _, fw_time, fw_iter, _, S = \
            projection_on_permutahedron_using_afw_euclidean(n, y, epsilon, lmo, {x_0: 1}, x_0)

        time_steps.append(sum(fw_time))

        fw_iterations.append(fw_iter)
        total_time = total_time + sum(fw_time)

    return time_steps, total_time, fw_iterations, step_regret


def isotonic_projection_mirror_descent(n, P, T, loss_vectors_list, x_0, eta):
    total_time = 0.0
    time_steps = []
    step_regret = []
    opt, alg = 0.0, 0.0

    def lmo(c):
        _, v = P.linear_optimization_over_base(c)
        return tuple(v)

    x = x_0
    for t in range(T):
        t1 = time.time()
        logging.warning('Iteration ' + str(t))

        l = loss_vectors_list[t]
        sol, x_star = P.linear_optimization_over_base(list(-l))

        opt = np.dot(l, x_star)
        alg = np.dot(l, x)
        step_regret.append(alg - opt)

        y = x - eta * l
        g = {i: P.f.g[i] for i in range(n + 1)}
        x = isotonic_projection(y, g)

        t2 = time.time()

        total_time = total_time + (t2 - t1)
        time_steps.append(t2 - t1)

    return time_steps, total_time, step_regret


def active_set_optimized_mirror_descent(n, epsilon, P, T, loss_vectors_list, x_0, eta):
    total_time = 0.0
    fw_iterations = []
    time_steps = []
    opt, alg = 0.0, 0.0
    step_regret = []

    def lmo(c):
        _, v = P.linear_optimization_over_base(c)
        return tuple(v)

    S = {x_0: 1}
    x = x_0
    for t in range(T):
        logging.warning('Iteration ' + str(t))
        # sigma = sample_vertex_from_convex_combination(S)
        sigma = x

        l = loss_vectors_list[t]
        loss = np.dot(l, sigma)
        alg = loss

        sol, x_star = P.linear_optimization_over_base(list(-l))
        opt = np.dot(l, x_star)
        step_regret.append(alg - opt)

        y = x - eta * l
        if t > 0:
            x, S = convex_hull_correction2(S, y)

        x, _, fw_time, fw_iter, _, S = \
            projection_on_permutahedron_using_afw_euclidean(n, y, epsilon, lmo, S, x)

        time_steps.append(sum(fw_time))

        fw_iterations.append(fw_iter)
        total_time = total_time + sum(fw_time)

    return time_steps, total_time, fw_iterations, step_regret


def cut_optimized_mirror_descent(n, epsilon, P, T, loss_vectors_list, x_0, eta):
    def get_chain(tight_sets):
        if frozenset(set(range(n))) not in tight_sets:
            tight_sets.add(frozenset(range(n)))
        if frozenset(set()) not in tight_sets:
            tight_sets.add(frozenset(set()))

        tight_sets = list(tight_sets)
        tight_sets.sort(key=len)

        for i in range(1, len(tight_sets)):
            if not tight_sets[i - 1].issubset(tight_sets[i]):
                raise ValueError('Tight sets list is bad :(')

        if tight_sets[-1] != frozenset(range(n)):
            tight_sets.append(frozenset(range(n)))

        return tight_sets

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

    total_time = 0.0
    fw_iterations = []
    points_list = []
    tight_sets_list = []
    opt, alg = 0.0, 0.0
    time_steps = []
    step_regret = []

    def lmo(c):
        _, v = P.linear_optimization_over_base(c)
        return tuple(v)

    S = {x_0: 1}
    x = x_0
    for t in range(T):
        logging.warning('Iteration ' + str(t))
        # sigma = sample_vertex_from_convex_combination(S)
        sigma = x

        l = loss_vectors_list[t]
        loss = np.dot(l, sigma)
        alg = loss

        sol, x_star = P.linear_optimization_over_base(list(-l))
        opt = np.dot(l, x_star)

        step_regret.append(alg - opt)

        y = x - eta * l
        points_list.append(y)

        tight_sets = set()

        t1 = time.time()
        for i in range(len(points_list) - 1):
            d = np.linalg.norm(y - points_list[i])
            tight_sets.union(infer_tight_sets_from_close_point(d, tight_sets_list[i]))
        t2 = time.time()

        chain_of_tight_sets = set(get_chain(tight_sets))

        def lmo(c):
            _, v = P.linear_optimization_tight_sets(c, chain_of_tight_sets)
            return v

        h = lambda x: 0.5 * np.dot(x - y, x - y)
        grad_h = lambda x: np.power(x - y, 1)
        h_tol, time_tol = -1, np.inf

        x, _, fw_time, fw_iter, _, S, _ = \
            adaptive_AFW_cardinality_polytope(x, {tuple(x): 1}, P, epsilon, h, grad_h, h_tol,
                                              time_tol,
                                              chain_of_tight_sets, y)

        time_steps.append(sum(fw_time) + t2 - t1)

        tight_sets_list.append(determine_tight_sets(y, x))
        fw_iterations.append(fw_iter)
        total_time = total_time + sum(fw_time)

    return time_steps, total_time, fw_iterations, step_regret


def doubly_optimized_mirror_descent(n, epsilon, P, T, loss_vectors_list, x_0, eta):
    def get_chain(tight_sets):
        if frozenset(set(range(n))) not in tight_sets:
            tight_sets.add(frozenset(range(n)))
        if frozenset(set()) not in tight_sets:
            tight_sets.add(frozenset(set()))

        tight_sets = list(tight_sets)
        tight_sets.sort(key=len)

        for i in range(1, len(tight_sets)):
            if not tight_sets[i - 1].issubset(tight_sets[i]):
                raise ValueError('Tight sets list is bad :(')

        if tight_sets[-1] != frozenset(range(n)):
            tight_sets.append(frozenset(range(n)))

        return tight_sets

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

    total_time = 0.0
    fw_iterations = []
    points_list = []
    tight_sets_list = []
    opt, alg = 0.0, 0.0
    time_steps = []
    step_regret = []

    def lmo(c):
        _, v = P.linear_optimization_over_base(c)
        return tuple(v)

    S = {x_0: 1}
    x = x_0
    for t in range(T):
        logging.warning('Iteration ' + str(t))

        # sigma = sample_vertex_from_convex_combination(S)
        sigma = x

        l = loss_vectors_list[t]
        loss = np.dot(sigma, l)
        alg = loss

        sol, x_star = P.linear_optimization_over_base(list(-l))
        opt = np.dot(l, x_star)
        step_regret.append(alg - opt)

        y = x - eta * l
        points_list.append(y)

        tight_sets = set()
        t1 = time.time()
        for i in range(len(points_list) - 1):
            d = np.linalg.norm(y - points_list[i])
            tight_sets.union(infer_tight_sets_from_close_point(d, tight_sets_list[i]))
        chain_of_tight_sets = set(get_chain(tight_sets))
        t2= time.time()

        def lmo(c):
            _, v = P.linear_optimization_tight_sets(c, chain_of_tight_sets)
            return v

        h = lambda x: 0.5 * np.dot(x - y, x - y)
        grad_h = lambda x: np.power(x - y, 1)
        h_tol, time_tol = -1, np.inf

        x, S = convex_hull_correction2(S, y)
        x, _, fw_time, fw_iter, _, S, _ = \
            adaptive_AFW_cardinality_polytope(x, S, P, epsilon, h, grad_h, h_tol,
                                              time_tol,
                                              chain_of_tight_sets, y)

        time_steps.append(sum(fw_time) + t2 - t1)

        tight_sets_list.append(determine_tight_sets(y, x))
        fw_iterations.append(fw_iter)
        total_time = total_time + sum(fw_time)

    return time_steps, total_time, fw_iterations, step_regret


def online_FW(x, lmo, T, loss_vectors_list, G=None):
    # record primal gap, function value, and time every iteration
    time = []
    grad_list = [np.zeros(len(x))]

    # initialize starting point and active set
    t = 0
    x_t = [x]
    v_t = {}
    loss = []

    # define blocksizes and random sampling parameters
    k = int(np.ceil(T ** (1 / 3)))

    if G:
        delta = 2 / (G * len(x) ** 0.5 * k ** 2)
    else:
        delta = 2 / (len(x) ** 1.5 * k ** 2)

    while t < T:

        start = datetime.datetime.now()

        if t % k != 0:

            # play x_{t-1}, i.e. dont do anything
            x_t.append(x_t[-1])

            # observe gradient/loss
            grad = loss_vectors_list[t]
            grad_list.append(grad)

            # compute loss
            loss.append(np.dot(x_t[-1], grad))

        else:
            v = []
            for i in range(k):
                v_ = np.random.normal(0, 1, n)
                v.append(v_ / np.linalg.norm(v_))
            v = np.array(v)
            grad_sum = np.sum(np.array(grad_list), axis=0)
            x = np.array([lmo(grad_sum + v[j] / delta) for j in range(k)])

            # play average of x
            x_t.append(np.mean(x, axis=0))

            # observe gradient/loss
            grad = loss_vectors_list[t]
            grad_list.append(grad)

            # compute loss
            loss.append(np.dot(x_t[-1], grad))

        end = datetime.datetime.now()
        time.append((end - start).total_seconds())

        t += 1

    return x_t, time, loss


def online_mirror_descent_permutahedron(P: Permutahedron, T: int, epsilon: float, seed, a, b):
    """
    Performs online mirror descent on a permutahedron
    :param P: permuathedron. See submodular_polytope.py for class definition
    :param T: number of iterations
    :param a: number of permutations
    :param b: swap distance between permutations
    :return: Total regret
    """
    n = len(P)
    D = (n ** 3 - n) / 6  # Diameter of permutahedron
    G = n  # Upper bound on norm l1 norm
    alpha = 1  # For Euclidean projection
    eta = (D / G) * math.sqrt((2 * alpha) / T)

    b = b // 2
    loss_vectors_list = generate_loss_functions_for_permutahedron(a, b, seed, n, T)
    x_0 = tuple(generate_random_permutation(n))

    def lmo(c):
        _, v = P.linear_optimization_over_base(-c)
        return v

    opt_values = []
    for t in range(T):
        x_star = lmo(loss_vectors_list[t])
        opt_values.append(np.dot(x_star, loss_vectors_list[t]))

    # Vanilla FW
    logging.warning('Starting vanilla OMD')
    time_steps_vanilla, total_time_vanilla, fw_iterations_vanilla, regret_vanilla = \
        vanilla_mirror_descent(n, epsilon, P, T, loss_vectors_list, x_0, eta)

    # Active set optimized FW
    logging.warning('Starting active set optimized OMD')
    time_steps_active_set_optimized, total_time_active_set_optimized, fw_iterations_active_set_optimized, regret_active_set_optimized = \
        active_set_optimized_mirror_descent(n, epsilon, P, T, loss_vectors_list, x_0, eta)

    # Cut optimized FW
    logging.warning('Starting cut optimized OMD')
    time_steps_cut_optimized, total_time_cut_optimized, fw_iterations_cut_optimized, regret_cut_optimized = \
        cut_optimized_mirror_descent(n, epsilon, P, T, loss_vectors_list, x_0, eta)

    # Doubly optimized FW
    logging.warning('Starting doubly optimized OMD')
    time_steps_doubly_optimized, total_time_doubly_optimized, fw_iterations_doubly_optimized, regret_doubly_optimized\
        = doubly_optimized_mirror_descent(n, epsilon, P, T, loss_vectors_list, x_0, eta)

    # Isotonic projection
    logging.warning('Starting isotonic projection OMD')
    time_steps_isotonic, total_time_isotonic, regret_isotonic = \
        isotonic_projection_mirror_descent(n, P, T, loss_vectors_list, x_0, eta)

    # Online FW
    logging.warning('Starting Online FW')
    iterates_ofw, time_steps_ofw, loss_ofw = online_FW(x_0, lmo, T, loss_vectors_list)
    total_time_ofw = sum(time_steps_ofw)
    regret_ofw = loss_ofw - np.array(opt_values)

    return total_time_vanilla, total_time_active_set_optimized, total_time_cut_optimized, \
        total_time_doubly_optimized, total_time_ofw, total_time_isotonic, \
        time_steps_vanilla, time_steps_active_set_optimized, time_steps_cut_optimized, \
        time_steps_doubly_optimized, time_steps_ofw, time_steps_isotonic, \
        fw_iterations_vanilla, fw_iterations_active_set_optimized, \
        fw_iterations_cut_optimized, fw_iterations_doubly_optimized, \
        regret_vanilla, regret_active_set_optimized, regret_cut_optimized, \
        regret_doubly_optimized, regret_ofw, regret_isotonic


n = 50
T = 1000
start = 10
end = 20
epsilon = math.pow(10, -3)
a, b = 6, 6

print('n =', n, 'T =', T, 'epsilon =', epsilon, 'a =', a, 'b =', b)

f = PermutahedronSubmodularFunction(n)
P = Permutahedron(f)
for seed in range(start, end):
    print('Run ', seed)
    t1, t2, t3, t4, t5, t6, T1, T2, T3, T4, T5, T6, i1, i2, i3, i4, r1, r2, r3, r4, r5, r6 =\
        online_mirror_descent_permutahedron(P, T, epsilon, seed, a, b)

    times = pd.DataFrame(columns=[T1, T2, T3, T4, T5, T6])
    iterates = pd.DataFrame(columns=[i1, i2, i3, i4])
    regrets = pd.DataFrame(columns=[r1, r2, r3, r4, r5, r6])
    times.to_csv('times_permutahedron' + str(n) + '_' + str(T) + '_' + str(a) + '_' + str(b) +
                 '_' + str(seed) + '.csv')
    iterates.to_csv('iterates_permutahedron' + str(n) + '_' + str(T) + '_' + str(a) + '_' +
                    str(b) + '_' + str(seed) + '.csv')
    regrets.to_csv('regrets_permutahedron' + str(n) + '_' + str(T) + '_' + str(a) + '_' + str(b) +
                   '_' + str(seed) + '.csv')

    sourceFile = open('permutahedron2_' + str(n) + '_' + str(T) + '_' + str(a) + '_' + str(b) +
                      '.txt', 'a')
    print('Iteration ' + str(seed), file=sourceFile)

    print('Total times', file=sourceFile)
    print(t1, file=sourceFile)
    print(t2, file=sourceFile)
    print(t3, file=sourceFile)
    print(t4, file=sourceFile)
    print(t5, file=sourceFile)
    print(t6, file=sourceFile)

    print('Regrets', file=sourceFile)
    print(sum(r1), file=sourceFile)
    print(sum(r2), file=sourceFile)
    print(sum(r3), file=sourceFile)
    print(sum(r4), file=sourceFile)
    print(sum(r5), file=sourceFile)
    print(sum(r6), file=sourceFile)

    sourceFile.close()
