import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

m = 500

# Iterations numbers of runs
runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
T_u, T_a, T_c, T_d = [], [], [], []

for i in runs:
    regret_df = pd.read_csv('iterates_' + str(i) + '.csv', index_col=0, header=None)
    regret_df.index = range(4)
    regret_df = regret_df.loc[[0, 1, 2, 3]]
    regret_df = (1000 * regret_df)/regret_df.sum(axis=1)[0]

    T_u.append(list(regret_df.loc[0]))
    T_a.append(list(regret_df.loc[1]))
    T_c.append(list(regret_df.loc[2]))
    T_d.append(list(regret_df.loc[3]))

T_u = pd.DataFrame(T_u)
T_a = pd.DataFrame(T_a)
T_c = pd.DataFrame(T_c)
T_d = pd.DataFrame(T_d)

T_u = T_u.cumsum(axis=1).to_numpy()
T_a = T_a.cumsum(axis=1).to_numpy()
T_c = T_c.cumsum(axis=1).to_numpy()
T_d = T_d.cumsum(axis=1).to_numpy()

T_u = pd.DataFrame(T_u)
T_a = pd.DataFrame(T_a)
T_c = pd.DataFrame(T_c)
T_d = pd.DataFrame(T_d)

matplotlib.rcParams.update({'font.size': 20})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

quantile25_u = T_u.quantile(0.25, axis=0)
quantile25_a = T_a.quantile(0.25, axis=0)
quantile25_c = T_c.quantile(0.25, axis=0)
quantile25_d = T_d.quantile(0.25, axis=0)

quantile75_u = T_u.quantile(0.75, axis=0)
quantile75_a = T_a.quantile(0.75, axis=0)
quantile75_c = T_c.quantile(0.75, axis=0)
quantile75_d = T_d.quantile(0.75, axis=0)

average_u = T_u.mean(axis=0)
average_a = T_a.mean(axis=0)
average_c = T_c.mean(axis=0)
average_d = T_d.mean(axis=0)

plt.title(r'AFW iterates for OMD-AFW variants')
plt.xlabel(r'Iterations')
plt.ylabel(r'Number of AFW iterates')

# plt.xscale('log')
# plt.yscale('log')

# plt.ylim(ymin=10, ymax=1200)
# plt.xlim(xmin=0, xmax=500)

plt.fill_between(range(m), quantile25_u, quantile75_u, color='green', alpha=0.1)
plt.fill_between(range(m), quantile25_a, quantile75_a, color='red', alpha=0.1)
plt.fill_between(range(m), quantile25_c, quantile75_c, color='blue', alpha=0.1)
plt.fill_between(range(m), quantile25_d, quantile75_d, color='orange', alpha=0.1)

average_u_plot, = plt.plot(average_u, color='green', label='OMD-UAFW')
average_a_plot, = plt.plot(average_a, color='red', label='OMD-ASAFW')
average_c_plot, = plt.plot(average_c, color='blue', label='OMD-TSAFW')
average_d_plot, = plt.plot(average_d, color='orange', label='OMD-A$^2$FW')

plt.tight_layout()

plt.legend(loc='lower right')
plt.savefig('iterates.png', dpi=300)
plt.show()

