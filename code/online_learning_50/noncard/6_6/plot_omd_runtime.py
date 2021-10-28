import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

m = 1000

# Iterations: indices of runs
runs = [1, 5, 6, 7, 8, 10, 11, 12, 15, 16, 17, 21, 23, 25, 26, 35, 36, 37, 38, 39]
T_u, T_a, T_c, T_d, T_o = [], [], [], [], []
T_i = []

for i in runs:
    # Read data from csv files
    time_df = pd.read_csv('times_50_1000_6_6_' + str(i) + '.csv', index_col=0,
                          header=None)
    time_df.index = range(5)
    time_df = time_df.loc[[0, 1, 2, 3, 4]]
    time_df = (1000 * time_df)/time_df.sum(axis=1)[0]

    T_u.append(list(time_df.loc[0]))
    T_a.append(list(time_df.loc[1]))
    T_c.append(list(time_df.loc[2]))
    T_d.append(list(time_df.loc[3]))
    T_o.append(list(time_df.loc[4]))
    # T_i.append(list(time_df.loc[5]))

T_u = pd.DataFrame(T_u)
T_a = pd.DataFrame(T_a)
T_c = pd.DataFrame(T_c)
T_d = pd.DataFrame(T_d)
T_o = pd.DataFrame(T_o)
# T_i = pd.DataFrame(T_i)

t_u = T_u.sum(axis=1).mean()
t_a = T_a.sum(axis=1).mean()
t_c = T_c.sum(axis=1).mean()
t_d = T_d.sum(axis=1).mean()
t_o = T_o.sum(axis=1).mean()
# t_i = T_i.sum(axis=1).mean()

T_u = T_u.cumsum(axis=1)
T_a = T_a.cumsum(axis=1)
T_c = T_c.cumsum(axis=1)
T_d = T_d.cumsum(axis=1)
T_o = T_o.cumsum(axis=1)
# T_i = T_i.cumsum(axis=1)

matplotlib.rcParams.update({'font.size': 20})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

quantile25_u = T_u.quantile(0.25, axis=0)
quantile25_a = T_a.quantile(0.25, axis=0)
quantile25_c = T_c.quantile(0.25, axis=0)
quantile25_d = T_d.quantile(0.25, axis=0)
quantile25_o = T_o.quantile(0.25, axis=0)
# quantile25_i = T_i.quantile(0.25, axis=0)

quantile75_u = T_u.quantile(0.75, axis=0)
quantile75_a = T_a.quantile(0.75, axis=0)
quantile75_c = T_c.quantile(0.75, axis=0)
quantile75_d = T_d.quantile(0.75, axis=0)
quantile75_o = T_o.quantile(0.75, axis=0)
# quantile75_i = T_i.quantile(0.75, axis=0)

average_u = T_u.mean(axis=0)
average_a = T_a.mean(axis=0)
average_c = T_c.mean(axis=0)
average_d = T_d.mean(axis=0)
average_o = T_o.mean(axis=0)
# average_i = T_i.mean(axis=0)

print(average_u[-1:], average_a[-1:], average_c[-1:], average_d[-1:], average_o[-1:])


plt.title(r'Runtime for OMD-AFW variants')
plt.xlabel(r'Iterations')
plt.ylabel(r'Running time (normzalied)')

# plt.xscale('log')
# plt.yscale('log')

# Uncomment to adjust axes ranges
# plt.ylim(ymin=5, ymax=1200)
# plt.xlim(xmin=0, xmax=500)

plt.fill_between(range(m), quantile25_u, quantile75_u, color='green', alpha=0.1)
plt.fill_between(range(m), quantile25_a, quantile75_a, color='red', alpha=0.1)
plt.fill_between(range(m), quantile25_c, quantile75_c, color='blue', alpha=0.1)
plt.fill_between(range(m), quantile25_d, quantile75_d, color='orange', alpha=0.1)
# plt.fill_between(range(m), quantile25_o, quantile75_o, color='cyan', alpha=0.1)
# plt.fill_between(range(m), quantile25_i, quantile75_i, color='brown', alpha=0.1)

average_u_plot, = plt.plot(average_u, color='green', linestyle='dotted', label='OMD-UAFW')
average_a_plot, = plt.plot(average_a, color='red', linestyle='dotted', label='OMD-ASAFW')
average_c_plot, = plt.plot(average_c, color='blue', linestyle='dashed', label='OMD-TSAFW')
average_d_plot, = plt.plot(average_d, color='orange', linestyle='dotted', label='OMD-A$^2$FW')
# average_o_plot, = plt.plot(average_o, color='cyan', linestyle='dashed', label='OFW')
# average_i_plot, = plt.plot(average_i, color='brown', linestyle='dotted', label='OMD-PAV')

plt.tight_layout()

plt.legend(loc='upper left', fontsize=15)
plt.savefig('runtime.png', dpi=300)
plt.show()
