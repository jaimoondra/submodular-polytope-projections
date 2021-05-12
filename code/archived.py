from online_learning import *


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


def projection_on_permutahedron_using_afw_hassan(n, y, epsilon, S={}, x=None):
    def f(s):
        return sum([n + 1 - i for i in range(1, s + 1)])

    g = {i: f(i) for i in range(1, n + 1)}
    g[0] = 0

    y = np.array(y)
    h = lambda x: 0.5 * np.dot(x - y, x - y)
    grad_h = lambda x: np.power(x - y, 1)
    lmo = lambda w: greedy_submodular_base(-w, g)
    h_tol, time_tol = -1, np.inf

    if x is None:
        w = generate_random_permutation(n)
        x = w
        S = {tuple(w): 1}

    AFW_sol, AFW_func, AFW_time, AFW_iter, AFW_gap, S, _ =\
        AFW(np.array(x), S, lmo , epsilon, h, grad_h, h_tol, time_tol)

    print(AFW_iter)
    return AFW_sol

