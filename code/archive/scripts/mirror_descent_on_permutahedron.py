import logging

logging.basicConfig(level=logging.WARNING)

from archive.continuous_methods import *
from archive.isotonic_projection import *
from archive.constants import *
from archive.submodular_polytope import PermutahedronSubmodularFunction


def sample_vertex_from_convex_combination(S):
    vertices = list(S.keys())
    s = len(S)
    distribution = np.array(list(S.values()))
    sum_distribution = sum(distribution)
    distribution = (1/sum_distribution) * distribution
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

    # if x is None:
    #     w = generate_random_permutation(n)
    #     x = w
    #     S = {tuple(w): 1}

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

        time_steps.append(sum(fw_time))

        tight_sets_list.append(determine_tight_sets(y, x))
        fw_iterations.append(fw_iter)
        total_time = total_time + sum(fw_time) + t2 - t1

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

        sigma = sample_vertex_from_convex_combination(S)

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
            adaptive_AFW_cardinality_polytope(x, {tuple(x): 1}, P, epsilon, h, grad_h, h_tol,
                                              time_tol,
                                              chain_of_tight_sets, y)

        time_steps.append(sum(fw_time))

        tight_sets_list.append(determine_tight_sets(y, x))
        fw_iterations.append(fw_iter)
        total_time = total_time + sum(fw_time) + t2 - t1

    return time_steps, total_time, fw_iterations, step_regret


def online_FW(x, lmo, T, loss_vectors_list, G=None):
    # record primal gap, function value, and time every iteration
    time = [0]
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
        time.append(time[t - 1] + (end - start).total_seconds())

        t += 1

    return x_t, time, loss


def online_mirror_descent_permutahedron(P: Permutahedron, T: int, epsilon: float):
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

    # a: number of permutations
    # b: number of swaps
    a, b = 1, 1
    loss_vectors_list = generate_loss_functions_for_permutahedron(a, b, 0, n, T)
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
        total_time_doubly_optimized, total_time_isotonic, total_time_ofw, \
        time_steps_vanilla, time_steps_active_set_optimized, time_steps_cut_optimized, \
        time_steps_doubly_optimized, time_steps_isotonic, time_steps_ofw, \
        fw_iterations_vanilla, fw_iterations_active_set_optimized, \
        fw_iterations_cut_optimized, fw_iterations_doubly_optimized, \
        regret_vanilla, regret_active_set_optimized, regret_cut_optimized, \
        regret_doubly_optimized, regret_isotonic, regret_ofw


n = 25
T = 200
epsilon = 1 / (n ** 3)

f = PermutahedronSubmodularFunction(n)
P = Permutahedron(f)


t1, t2, t3, t4, t5, t6, T1, T2, T3, T4, T5, T6, i1, i2, i3, i4, r1, r2, r3, r4, r5, r6 =\
    online_mirror_descent_permutahedron(P, T, epsilon)

print('n = 10, T = 500, Epsilon = 1/n^2')

print(t1)
print(t2)
print(t3)
print(t4)
print(t5)
print(t6)

print(T1)
print(T2)
print(T3)
print(T4)
print(T5)
print(T6)

print(i1)
print(i2)
print(i3)
print(i4)

print(r1)
print(r2)
print(r3)
print(r4)
print(r5)
print(r6)

print(sum(r1))
print(sum(r2))
print(sum(r3))
print(sum(r4))
print(sum(r5))
print(sum(r6))
