from archive.isotonic_projection import *
from archive.submodular_polytope import PermutahedronSubmodularFunction, CardinalityPolytope


y = np.array([ 20.84 ,  2.81, 106.81,  82.01,  89.67,  42.09, 25.14,  62.26,  52.9,   45.45,
  27.57, 114.61,   2.08,  55.9,   26.95,  29.81,   0.96,  58.75,  37.39,   0.45,
  43.91,   7.82,  12.83,  49.44,  16.94,  11.81,  31.88,  59.38,  71.06,   7.67,
  13.45, 111.57, 121.74,   5.64,  18.52,  67.98,  25.09,  42.21,   0,    27.12,
  15.68,  38.55,  93.4,   86.56,  73.38,  16.78,  30.57,   2.4,   41.46,   4.39])


f = PermutahedronSubmodularFunction(50)
P = CardinalityPolytope(f)
g = P.f.g
g = {i: g[i] for i in range(51)}
x = isotonic_projection(y, g)
# print(x)

S = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 48}
x_s = 0
for i in S:
    x_s = x_s + x[i]

# print(x_s)
determine_tight_sets(y, x, g)
