import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


m = 500
seen = pd.read_csv(r'seen.csv')
learned = pd.read_csv(r'learned.csv')

print(seen, learned)

matplotlib.rcParams.update({'font.size': 15})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

quantile10_learned = learned.quantile(0.15, axis=1)
quantile10_seen = seen.quantile(0.15, axis=1)
quantile90_learned = learned.quantile(0.85, axis=1)
quantile90_seen = seen.quantile(0.85, axis=1)
average_learned = learned.mean(axis=1)
average_seen = seen.mean(axis=1)

plt.title(r'Tight sets from close points for $n = 100$, noise $1/50$')
plt.xlabel(r'Iterations')
plt.ylabel(r'Fraction of tight sets for close points')

plt.fill_between(range(m), quantile10_seen, quantile90_seen, color='green', alpha=0.1)
plt.fill_between(range(m), quantile10_learned, quantile90_learned, color='blue', alpha=0.1)
seen, = plt.plot(average_seen, color='green', label='Fraction of tight sets seen in previous '
                                                    'points')
learned, = plt.plot(average_learned, color='blue', label='Fraction of tight sets recoverd by '
                                                         'Theorem 5')

plt.legend(loc='lower right')
plt.savefig('tight_sets_close_points.png', dpi=300)
plt.show()

