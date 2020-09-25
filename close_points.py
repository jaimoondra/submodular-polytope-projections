from typing import List
import numpy as np
import math
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from inc_fix import CardinalityPolytope, IncFix
from constants import *


logging.basicConfig(level=logging.WARNING)


def swati_thesis_example():
    g = [0.5, 0.6, 0.7, 0.8, 0.87]
    y = [1, 0.9, 0.9, 0.9, 1.1]
    proj = IncFix(g, y)
    x_star, iterates = proj.projection()
    print(x_star, iterates)


def generate_concave_function(seed=0):
    if seed is not None:
        np.random.seed(0)

    g = random_point(n, 1, True, seed)
    g = [round(x, base_decimal_accuracy) for x in g]
    g.sort(reverse=True)

    for i in range(1, len(g)):
        g[i] = g[i - 1] + g[i]

    g = round_list(g, base_decimal_accuracy)

    return g


def round_list(x: List[float], accuracy: int = base_decimal_accuracy):
    x = [round(i, accuracy) for i in x]
    return x


def plot_iterates(iterates: List[List[float]], y: List[float], color1, color2):
    jump = max(int(n / 10), min(5, int(n / 2)))
    for i in range(len(iterates)):
        iterates[i] = round_list(iterates[i], base_decimal_accuracy)
        if i % jump == 1 or i == len(iterates) - 1:
            plt.plot(iterates[i], color1)

    plt.xlabel('Ground set elements')
    plt.ylabel('Value')
    plt.plot(y, color2)


def plot_tight_sets(tight_sets: List[List[int]], color1):
    plt.xlabel('Iteration')
    plt.ylabel('Tight set size')
    plt.plot(tight_sets, color1)


def random_point(n: int, r: float = 1, nonnegative: bool = False, seed = 0):
    if seed is not None:
        np.random.seed(seed)

    z = np.random.multivariate_normal(mean=[0] * n, cov=r * np.identity(n))
    if nonnegative:
        z = [x if x >= 0 else -x for x in z]

    return z


# Cardinality function a.k.a positive concave function
seed = seeds[0]
g = generate_concave_function(seed)
print('g = ' + str(g))

# First random point
seed = seeds[1]
np.random.seed(seed)
y0 = random_point(n, 1.5, True, seed)
y0 = [round(i, base_decimal_accuracy) for i in y0]
y0.sort(reverse=True)
print('y0 = ' + str(y0))

proj = IncFix(g, y0)
x_star0, iterates0, tight_sets0 = proj.projection()
x_star0 = round_list(x_star0, base_decimal_accuracy)
print('projection of y0 on P(f) = ' + str(x_star0))

# Point close to first random point
noise = random_point(n, (1 / (n * n)), False, seed)
y1 = [round(y0[i] + noise[i], base_decimal_accuracy) for i in range(n)]
y1.sort(reverse=True)
print('y1 = ' + str(y1))

proj = IncFix(g, y1)
x_star1, iterates1, tight_sets1 = proj.projection()
x_star1 = round_list(x_star1, base_decimal_accuracy)
print('projection of y1 on P(f) = ' + str(x_star1))

# First random point
seed = seeds[2]
np.random.seed(seed)
y2 = random_point(n, 1.5, True, seed=seed)
y2 = [round(i, base_decimal_accuracy) for i in y2]
print('y2 = ' + str(y2))

proj = IncFix(g, y2)
x_star2, iterates2, tight_sets2 = proj.projection()
x_star2 = round_list(x_star2, base_decimal_accuracy)
print('projection of y2 on P(f) = ' + str(x_star2))

# Plots - close points
plt.title('Iterates with time - 2 close points')
plot_iterates(iterates0, y0, 'b', 'c')
plot_iterates(iterates1, y1, 'r', 'm')
plt.savefig('figures\\' + str(n) + '-close-points-iterates.png')
plt.clf()

plt.title('Size of tight sets with time - 2 close points')
plot_tight_sets(tight_sets0, 'b')
plot_tight_sets(tight_sets1, 'r')
plt.savefig('figures\\' + str(n) + '-close-points-tight-sets.png')
plt.clf()

# Plots - two random points
plt.title('Iterates with time - 2 random points')
plot_iterates(iterates0, y0, 'b', 'c')
plot_iterates(iterates2, y2, 'g', 'y')
plt.savefig('figures\\' + str(n) + '-random-points-iterates.png')
plt.clf()

plt.title('Size of tight sets with time - 2 random points')
plot_tight_sets(tight_sets0, 'b')
plot_tight_sets(tight_sets2, 'g')
plt.savefig('figures\\' + str(n) + '-random-points-tight-sets.png')
plt.clf()
