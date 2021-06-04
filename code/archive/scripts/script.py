# n = 50
# inner = 500
# outer = 20
#
# with open('n-50-1-over-n-noise-500-inner-20-outer.csv', mode='w') as write_file:
#     writer = csv.writer(write_file, delimiter=',')
#     writer.writerow(['Seed', 'Learned tight sets', 'Total tight sets', 'Fraction of tight sets '
#                                                                        'seen'])
#     for j in range(outer):
#         print('Outer iteration #' + str(j))
#         seens = close_points_learning(n=n, m=inner, seed=seeds[j])
#         for i in range(len(seens[0])):
#             writer.writerow([seeds[j], seens[0][i], seens[1][i], seens[0][i]/seens[1][i]])
#
#


from archive.submodular_polytope import CardinalitySubmodularFunction, CardinalityPolytope

f = CardinalitySubmodularFunction(g=[1.0, 1.9, 2.6], n=3)
P = CardinalityPolytope(f=f)
print(P.linear_optimization_tight_sets(c=[1.0, 1.1, 2.5], T=[set(), {0, 2}, {0, 1, 2}]))
