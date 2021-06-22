import logging
logging.basicConfig(level=logging.WARNING)

from archive.isotonic_projection import *
from archive.utils import *
import matplotlib.pyplot as plt
import matplotlib
from archive.submodular_polytope import PermutahedronSubmodularFunction, CardinalityPolytope
import pandas as pd


def close_points_learning(n: int, m: int = 40, seed=None, g=None, epsilon=0.1):
    """
    Uses algorithm described after Lemma 7 to determine save in running time
    :param n: Size of the ground set
    :param m: Number of points in the neighborhood
    :param seed: seed for random
    """
    np.random.seed(seed)

    def generate_close_random_point(y0):
        # print(n, epsilon)
        error = random_error(n=n, r=epsilon, seed=None, decial_accuracy=6)
        y = np.array([(y0[i] + error[i]) for i in range(n)])
        return y

    def seen_tight_sets_count(tight_sets0, tight_sets1):
        """
        Infers tights sets for tilde(y) from tight sets for y
        :param d: distance d(y, tilde(y))
        :param tight_sets1: tight_sets of point y, along with their c[j + 1] - c[j] values,
        sorted by their c[j + 1] - c[j] values
        :return: inferred tight sets for tilde(y), list
        """
        inferred_tight_sets = set()

        tight_sets_list0 = {tight_sets0[t][1] for t in range(len(tight_sets0))}
        tight_sets_list1 = {tight_sets1[t][1] for t in range(len(tight_sets1))}
        # print(tight_sets_list0)
        # print(tight_sets_list1)
        # print(tight_sets_list0.difference(tight_sets_list1))

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
        :param tight_sets_actual:
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
    y0 = np.array(random_point(n, t, c, nonnegative=False, seed=seed))
    print(y0)
    points.append(y0)

    # Calculate projection
    x0 = np.array(isotonic_projection(y0, g))
    projections.append(x0)
    tight_sets_list.append(determine_tight_sets(y0, x0, g))

    learned_tight_sets = [0]
    seen_tight_sets = [0]
    total_tight_sets = [len(tight_sets_list[0])]
    tight_sets = tight_sets_list[0]
    # print({tight_sets[t][1] for t in range(len(tight_sets))})

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

        # print({tight_sets[t][1] for t in range(len(tight_sets))})

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
m = 50
f = PermutahedronSubmodularFunction(n)
P = CardinalityPolytope(f)
outer = 50

matplotlib.rcParams.update({'font.size': 15})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
        close_points_learning(n, m, seed=2*t + 1, g=g, epsilon=1/(5 * math.sqrt(n)))

    df_learned['Run ' + str(t)] = fractional_learned_tight_sets
    df_seen['Run ' + str(t)] = fractional_seen_tight_sets


df_learned.to_csv(r'learned_n_50_noise_n.csv')
df_seen.to_csv(r'seen_n_50_noise_n.csv')
quantile10_learned = df_learned.quantile(0.15, axis=1)
quantile10_seen = df_seen.quantile(0.15, axis=1)
quantile90_learned = df_learned.quantile(0.85, axis=1)
quantile90_seen = df_seen.quantile(0.85, axis=1)
average_learned = df_learned.mean(axis=1)
average_seen = df_seen.mean(axis=1)


print(quantile90_seen, quantile90_learned, quantile10_seen, quantile10_learned)


plt.title(r'Tight sets from close points for $n = 100$, noise $1/50$')
plt.xlabel(r'Iterations')
plt.ylabel(r'Fraction of tight sets for close points')
plt.savefig('tight_cuts_seen_using_gradient_for_close_points_n_100_error_n.png', dpi=300)
plt.fill_between(range(m), quantile10_seen, quantile90_seen, color='green',
                 alpha=0.1)
plt.fill_between(range(m), quantile10_learned, quantile90_learned, color='blue',
                 alpha=0.1)
learned, = plt.plot(average_learned, color='blue')
seen, = plt.plot(average_seen, color='green')

plt.show()
