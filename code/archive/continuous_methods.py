from isotonic_projection import *
import numpy.linalg as npl
import datetime
from gurobipy import *
from time import process_time
from scipy.optimize import minimize, minimize_scalar
import logging
# from submodular_polytope import CardinalitySubmodularFunction, CardinalityPolytope, \
#     PermutahedronSubmodularFunction, Permutahedron
# from utils import *
# from constants import *
import random
import time
from rounding import *

logging.basicConfig(level=logging.INFO)


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

    def update_S(S, gamma, Away, vertex):
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
        T = {k: v for k, v in S.items() if np.round(v, 10) > 0}
        t = sum(list(T.values()))
        T = {k: T[k]/t for k in T}
        x = np.zeros(n)
        for k in T:
            x = x + np.array(k) * T[k]
        return T

        # return {k: v for k, v in S.items() if np.round(v, 12) > 0}

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
        if np.dot(-grad, d_FW) <= epsilon:
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
        x, gamma = line_search(x, d, gamma_max, func)
        # x, gamma = segment_search(func, grad_f, x, x + gamma_max *d)
        # update active set
        S = update_S(S, gamma, Away, vertex)
        end = process_time()
        time.append(time[t] + end - start)
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        t += 1

    return x, function_value, time, t, primal_gap, S


def adaptive_AFW_cardinality_polytope(x, S, P, epsilon, func, grad_f, f_tol, time_tol,
                                      initial_tight_sets, y):
    # Fucntion to compute away vertex
    n = len(x)
    def away_step(grad, S):
        costs = {}

        for k, v in S.items():
            cost = np.dot(k, grad)
            costs[cost] = [k, v]
        vertex, alpha = costs[max(costs.keys())]
        return vertex, alpha

    def update_S(S, gamma, Away, vertex):
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
        T = {k: v for k, v in S.items() if np.round(v, 10) > 0}
        t = sum(list(T.values()))
        T = {k: T[k]/t for k in T}
        x = np.zeros(n)
        for k in T:
            x = x + np.array(k) * T[k]
        return T

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
        start = process_time()              # Start time
        grad = grad_f(x)                    # Compute gradient

        # solve linear subproblem and compute FW direction
        def lmo(c, inferred_tight_sets):
            inferred_tight_sets_list = list(inferred_tight_sets)
            inferred_tight_sets_list.sort(key=len)
            _, v = P.linear_optimization_tight_sets(c, inferred_tight_sets_list)
            return v

        v = lmo(-grad, list(inferred_tight_sets))
        d_FW = v - np.array(x)
        gap = np.dot(-grad, d_FW)

        if gap < 0:
            gap = 0

        d = 2 * math.sqrt(gap)
        tight_sets = determine_tight_sets(y, x)
        new_inferred_tight_sets = infer_tight_sets_from_close_point(d, tight_sets)
        if not new_inferred_tight_sets.issubset(inferred_tight_sets):
            inferred_tight_sets = inferred_tight_sets.union(new_inferred_tight_sets)
            # x = lmo(-grad, list(inferred_tight_sets))
            try:
                x = cut_rounding(P, list(inferred_tight_sets), y, S)
                end = process_time()
                time.append(time[t] + end - start)
                f_improv = function_value[-1] - func(x)
                function_value.append(func(x))
                t += 1
                return x, function_value, time, t, primal_gap, S, inferred_tight_sets
            except ValueError:
                pass

        if gap <= epsilon:                      # If primal gap is small enough - terminate
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
        x, gamma = line_search(x, d, gamma_max, func)

        # update active set
        S = update_S(S, gamma, Away, vertex)
        end = process_time()
        time.append(time[t] + end - start)
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        t += 1

    return x, function_value, time, t, primal_gap, S, inferred_tight_sets


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


def projection_on_permutahedron_using_pv(n, y):
    f = PermutahedronSubmodularFunction(n)
    P = CardinalityPolytope(f)
    g = {i: P.f.g[i] for i in range(n + 1)}
    return isotonic_projection(np.array(y), g)

# n = 100
# random.seed(0)
# np.random.seed(0)
#
# for j in range(20):
#     x = np.random.permutation(n) + 1
#     S = {tuple(x): 1}
#     y = np.round(np.random.uniform(-n, n, n), 4)
#
#     epsilon = 1/math.pow(n, 1)
#
#     x1 = projection_on_permutahedron_using_pv(n, y)
#     t1 = time.time()
#     x2 = projection_on_permutahedron_using_afw(n, y, epsilon, S, x)
#     t2 = time.time()
#
#     print('time =', t2 - t1)
#     print('norm =', np.linalg.norm(x1 - x2))
