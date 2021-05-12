import numpy.linalg as npl
from scipy.optimize import minimize_scalar, minimize
import datetime
from gurobipy import *
from time import process_time
from scipy.optimize import minimize
import logging
from utils import *
from constants import *
from submodular_polytope import CardinalitySubmodularFunction, CardinalityPolytope, \
    PermutahedronSubmodularFunction, Permutahedron
from utils import generate_random_permutation
import random
import time


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


# Fucntion to compute away vertex
def away_step(grad, S):
    costs = {}

    for k, v in S.items():
        cost = np.dot(k, grad)
        costs[cost] = [k, v]
    vertex, alpha = costs[max(costs.keys())]
    return vertex, alpha


# Function to update active set
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
    return {k: v for k, v in S.items() if np.round(v, 6) > 0}


def line_search(x, d, gamma_max, func):
    def fun(gamma):
        ls = x + gamma * d
        return func(ls)

    res = minimize_scalar(fun, bounds=(0, gamma_max), method='bounded')

    gamma = res.x
    ls = x + gamma * d
    return ls, gamma


def AFW(x, S, lmo, epsilon,func,grad_f, f_tol, time_tol):
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
        v = lmo(grad)
        d_FW = v-x
        # If primal gap is small enough - terminate
        if np.dot(-grad, d_FW) <= epsilon:
            break
        else:
            # update convergence data
            primal_gap.append(np.dot(-grad,d_FW))
        # Compute away vertex and direction
        a, alpha_a = away_step(grad, S)
        d_A = x - a
        # Check if FW gap is greater than away gap
        if np.dot(-grad,d_FW) >= np.dot(-grad,d_A):
            # choose FW direction
            d = d_FW
            vertex = v
            gamma_max = 1
            Away = False
        else:
            # choose Away direction
            d = d_A
            vertex = a
            gamma_max = alpha_a/(1-alpha_a)
            Away = True
        # Update next iterate by doing a feasible line-search
        x, gamma = line_search(x, d, gamma_max,func)
        # x, gamma = segment_search(func, grad_f, x, x + gamma_max *d)
        # update active set
        S = update_S(S,gamma, Away, vertex)
        end = process_time()
        time.append(time[t] + end - start)
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        t += 1
        # if t == 1000:
        #     break
    return x, function_value, time, t, primal_gap, S


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
    m.write('exact.lp')
    m.optimize()
    return np.array([i.x for i in x]), np.array([lam[i].x for i in lam])


def maximal_tight(y, g):
    # sort y so we can check feasiblity in the base polytope
    pi = np.argsort(-y) + 1

    # find cummulative sums of sorted vector so we can check feasibility/violations
    s = np.cumsum(sorted(y, reverse=True))
    violations = np.round(np.array([g[i + 1] - j for i, j in enumerate(s)]), 6)

    if any(violations < 0):
        return 'y not feasible'
    elif all(violations > 0):
        return [0]
    else:
        return pi[:np.arange(len(y))[violations == 0][-1] + 1]


def chi(M):
    chi_0 = np.zeros(n)
    for i in M:
        chi_0[i - 1] = 1
    return chi_0


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


def sort_vector(v):
    return np.array([v[k] for k in sorted(list(v.keys()))])


def construct_function(n, g):
    ground_set = list(range(1, n + 1))
    discrete_concave = sorted(g, reverse=True)
    h = {}
    h[0] = 0
    for i, j in enumerate(discrete_concave):
        h[i + 1] = h[i] + j
    return h


def submodular_function(n, g):
    func = construct_function(n, g)
    ground = list(range(1, n + 1))

    def findsubsets(s, l):
        return list(itertools.combinations(s, l))

    subsets = []
    for i in range(1, n + 1):
        subsets.extend(findsubsets(ground, i))

    f = {}
    f[tuple([0])] = func[0]
    for i in subsets:
        f[i] = func[len(i)]

    return f


def submodular_oracle(S, func, card):
    if card == True:
        return func[len(S)]
    else:
        return func[S]


def greedy_submodular(w, func, card):
    # find permuation corresponding to cost vector sorted in decreasing order
    pi = np.argsort(-w) + 1

    # s is the optimal chain of elements in ground set and x is the corresponsing optimal solution constructed by greedy
    x = {}
    s = {}
    s[0] = []
    for i, j in enumerate(pi):
        # extend chain based on permuation above
        s[i + 1] = sorted(pi[:i + 1])

        # x is then the marginal gain
        if w[j - 1] > 0:
            x[j] = submodular_oracle(tuple(s[i + 1]), func, card) - submodular_oracle(tuple(s[i]),
                                                                                      func, card)
        else:
            x[j] = 0

    return sort_vector(x)


def greedy_submodular_base(w, func, card=True):
    # find permuation corresponding to cost vector sorted in decreasing order
    pi = np.argsort(-w) + 1

    # s is the optimal chain of elements in ground set and x is the corresponsing optimal
    # solution constructed by greedy
    x = {}
    s = {}
    s[0] = [0]
    for i, j in enumerate(pi):
        # extend chain based on permuation above
        s[i + 1] = sorted(pi[:i + 1])

        # x is then the marginal gain
        x[j] = submodular_oracle(tuple(s[i + 1]), func, card) - submodular_oracle(tuple(s[i]), func,
                                                                                  card)

    return sort_vector(x)


def greedy_submodular_card(w,func):
    card = True
    #find permuation corresponding to cost vector sorted in decreasing order
    pi = np.argsort(-w)+1
    #s is the optimal chain of elements in ground set and x is the corresponsing optimal solution constructed by gredy
    x = {}
    s = {}
    s[0] = []
    for i,j in enumerate(pi):
        #extend chain
        s[i+1] = sorted(pi[:i+1])
        #x is then the marginal gain
        x[j] = submodular_oracle(s[i+1],func,card) - submodular_oracle(s[i],func,card)
    return sort_vector(x)


def generate_loss_function_vector_for_permutahedron(n: int):
    x = [random.random() for i in range(n)]  # Each entry is random number between 0 and 1
    s = sum(x)
    x = [(x[i] * n) / s for i in range(n)]  # Normalize so the sum is always n
    x = round(x, 3)
    x.sort(reverse=True)
    return np.array(x)


def generate_losses_and_random_vector(n: int, T: int):
    loss_vectors_list = [generate_loss_function_vector_for_permutahedron(n) for i in range(T)]
    x = tuple(generate_random_permutation(n))
    return loss_vectors_list, x


def online_mirror_descent_permutahedron(P: Permutahedron, T: int):
    """
    Performs online mirror descent on a permutahedron
    :param P: permuathedron. See submodular_polytope.py for class definition
    :param T: number of iterations
    :return: Total regret
    """
    n = len(P)
    D = (n ** 3 - n) / 6  # Diameter of permutahedron
    G = n  # Upper bound on norm l1 norm
    alpha = 1  # For Euclidean projection
    eta = (D / G) * math.sqrt((2 * alpha) / T)

    total_time_vanilla, total_time_active_set_optimized, total_time_doubly_optimized = 0.0, 0.0, 0.0
    fw_iterations_vanilla, fw_iterations_active_set_optimized, fw_iterations_doubly_optimized = [], [], []

    loss_vectors_list, x_0 = generate_losses_and_random_vector(n, T)
    # Active vertex set with its coefficients
    S_vanilla, S_active_set_optimized, S_doubly_optimized = {x_0: 1}, {x_0: 1}, {x_0: 1}

    h_tol, time_tol, epsilon = -1, np.inf, 1 / n
    h = lambda z: 0.5 * np.dot(z - y, z - y)
    grad_h = lambda z: np.array(z) - np.array(y)

    def lmo(c):
        _, v = P.linear_optimization_over_base(c)
        return tuple(v)

    g = {i: P.f.g[i] for i in range(n + 1)}

    # Vanilla FW
    logging.info('Starting vanilla OMD')
    for t in range(T):
        logging.info('Iteration ' + str(t))
        max_coeff = 0
        x = x_0
        for sigma in S_vanilla:
            if S_vanilla[sigma] > max_coeff:
                max_coeff = S_vanilla[sigma]
                x = sigma

        l = loss_vectors_list[t]
        loss = np.dot(x, l)
        y = x - eta * l

        fw_sol, fw_func, fw_time, fw_iter, fw_gap, S_vanilla, _ = \
            AFW(x_0, {x_0: 1}, lmo, epsilon, h, grad_h, h_tol, time_tol)

        fw_iterations_vanilla.append(fw_iter)
        total_time_vanilla = total_time_vanilla + sum(fw_time)

    # Active set optimized FW
    logging.info('Starting active set optimized OMD')
    for t in range(T):
        logging.info('Iteration ' + str(t))
        max_coeff = 0
        x = x_0
        for sigma in S_active_set_optimized:
            if S_active_set_optimized[sigma] > max_coeff:
                max_coeff = S_active_set_optimized[sigma]
                x = sigma

        l = loss_vectors_list[t]
        loss = np.dot(x, l)
        y = x - eta * l

        fw_sol, fw_func, fw_time, fw_iter, fw_gap, S_active_set_optimized, _ = \
            AFW(x, S_active_set_optimized, lmo, epsilon, h, grad_h, h_tol, time_tol)

        fw_iterations_active_set_optimized.append(fw_iter)
        total_time_active_set_optimized = total_time_active_set_optimized + sum(fw_time)

    # Doubly optimized FW
    logging.info('Starting doubly optimized OMD')
    for t in range(T):
        logging.info('Iteration ' + str(t))
        max_coeff = 0
        x = x_0
        for sigma in S_doubly_optimized:
            if S_doubly_optimized[sigma] > max_coeff:
                max_coeff = S_doubly_optimized[sigma]
                x = sigma

        l = loss_vectors_list[t]
        loss = np.dot(x, l)
        y = x - eta * l

        x, S_doubly_optimized = convex_hull_correction2(S_doubly_optimized, y)
        fw_sol, fw_func, fw_time, fw_iter, fw_gap, S_doubly_optimized, _ = \
            AFW(x, S_doubly_optimized, lmo, epsilon, h, grad_h, h_tol, time_tol)

        fw_iterations_doubly_optimized.append(fw_iter)
        total_time_doubly_optimized = total_time_doubly_optimized + sum(fw_time)

    return total_time_vanilla, total_time_active_set_optimized, total_time_doubly_optimized, \
        fw_iterations_vanilla, fw_iterations_active_set_optimized, fw_iterations_doubly_optimized


def projection_on_permutahedron_using_afw(n, y, epsilon, S={}, x=None):
    """
    :param n: |E|
    :param y: Point to project
    :param epsilon: Error
    :param S: Ative set dict
    :param x: Intial iterate
    :return:
    """
    f = PermutahedronSubmodularFunction(n)
    P = CardinalityPolytope(f)

    y = np.array(y)
    h = lambda x: 0.5 * np.dot(x - y, x - y)
    grad_h = lambda x: np.power(x - y, 1)
    lmo = lambda w: P.linear_optimization_over_base(-np.array(w))[1]
    h_tol, time_tol = -1, np.inf

    if x is None:
        w = generate_random_permutation(n)
        x = w
        S = {tuple(w): 1}

    AFW_sol, AFW_func, AFW_time, AFW_iter, AFW_gap, S =\
        AFW(np.array(x), S, lmo, epsilon, h, grad_h, h_tol, time_tol)

    return AFW_sol


def projection_on_permutahedron_using_afw_hassan(n, y, epsilon, S={}, x=None):
    def f(s):
        return sum([n + 1 - i for i in range(1, s + 1)])

    g = {i: f(i) for i in range(1, n + 1)}
    g[0] = 0

    y = np.array(y)
    h = lambda x: 0.5 * np.dot(x - y, x - y)
    grad_h = lambda x: np.power(x - y, 1)
    lmo = lambda w: greedy_submodular_card(-w, g)
    h_tol, time_tol = -1, np.inf

    if x is None:
        w = generate_random_permutation(n)
        x = w
        S = {tuple(w): 1}

    AFW_sol, AFW_func, AFW_time, AFW_iter, AFW_gap, S =\
        AFW(np.array(x), S, lmo , epsilon, h, grad_h, h_tol, time_tol)

    # for g in AFW_gap:
    #     print(g)
    return AFW_sol


def projection_on_permutahedron_using_pv(n, y):
    f = PermutahedronSubmodularFunction(n)
    P = CardinalityPolytope(f)
    g = {i: P.f.g[i] for i in range(n + 1)}
    return isotonic_projection(np.array(y), g)


n = 100
random.seed(0)
np.random.seed(0)

for j in range(20):
    x = np.random.permutation(n) + 1
    S = {tuple(x): 1}
    y = np.round(np.random.uniform(-n, n, n), 4)

    epsilon = 1/math.pow(n, 1)

    x1 = projection_on_permutahedron_using_pv(n, y)
    t1 = time.time()
    x2 = projection_on_permutahedron_using_afw(n, y, epsilon, S, x)
    t2 = time.time()

    print('time =', t2 - t1)
    print('norm =', np.linalg.norm(x1 - x2))
