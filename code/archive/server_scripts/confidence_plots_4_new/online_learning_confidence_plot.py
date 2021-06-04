import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

m = 500

runs = [20, 25, 50, 51, 52, 53, 54, 70, 71, 72, 73, 74, 75, 80, 81, 100, 101, 102, 120, 121, 122]

T_u, T_a, T_c, T_d = [], [], [], []

for i in runs:
    time_df = pd.read_csv('iterates_4_' + str(i) + '.csv', index_col=0, header=None)
    time_df.index = range(4)
    time_df = time_df.loc[[0, 1, 2, 3]]
    # time_df = (1000 * time_df)/time_df.sum(axis=1)[0]

    print(i)
    print(time_df.sum(axis=1))

    T_u.append(list(time_df.loc[0]))
    T_a.append(list(time_df.loc[1]))
    T_c.append(list(time_df.loc[2]))
    T_d.append(list(time_df.loc[3]))

T_u = pd.DataFrame(T_u)
T_a = pd.DataFrame(T_a)
T_c = pd.DataFrame(T_c)
T_d = pd.DataFrame(T_d)

t_u = T_u.sum(axis=1).mean()
t_a = T_a.sum(axis=1).mean()
t_c = T_c.sum(axis=1).mean()
t_d = T_d.sum(axis=1).mean()

print(t_u, t_a, t_c, t_d)

#
# T_u, T_a, T_c, T_d, T_o, T_p = [], [], [], []
#
# for i in runs:
#     time_df = pd.read_csv('times_4_' + str(i) + '.csv', index_col=0, header=None)
#     time_df.index = range(6)
#     time_df = time_df.loc[[0, 1, 2, 3, 4, 5]]
#     time_df = (1000 * time_df)/time_df.sum(axis=1)[0]
#
#     print(i)
#     print(time_df.sum(axis=1))
#
#     T_u.append(list(time_df.loc[0]))
#     T_a.append(list(time_df.loc[1]))
#     T_c.append(list(time_df.loc[2]))
#     T_d.append(list(time_df.loc[3]))
#     T_p.append(list(time_df.loc[4]))
#     T_o.append(list(time_df.loc[5]))
#
# T_u = pd.DataFrame(T_u)
# T_a = pd.DataFrame(T_a)
# T_c = pd.DataFrame(T_c)
# T_d = pd.DataFrame(T_d)
# T_p = pd.DataFrame(T_o)
#
T_u = T_u.cumsum(axis=1)
T_a = T_a.cumsum(axis=1)
T_c = T_c.cumsum(axis=1)
T_d = T_d.cumsum(axis=1)

matplotlib.rcParams.update({'font.size': 16})
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

plt.title(r'Number of AFW iterations for OMD-AFW variants')
plt.xlabel(r'Iterations')
plt.ylabel(r'Number of AFW iterations')

# plt.xscale('log')
# plt.yscale('log')

# plt.ylim(ymin=5, ymax=1200)
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

plt.legend(loc='upper left', fontsize=15)
plt.savefig('iterations.png', dpi=300)
plt.show()


#
# for i in runs:
#     time_df = pd.read_csv('regrets_4_' + str(i) + '.csv', index_col=0, header=None)
#     time_df.index = range(6)
#     time_df = time_df.loc[[0, 1, 2, 3, 4, 5]]
#     # time_df = (1000 * time_df)/time_df.sum(axis=1)[0]
#
#     print(i)
#     print(time_df.sum(axis=1))
#
#     T_u.append(list(time_df.loc[0]))
#     T_a.append(list(time_df.loc[1]))
#     T_c.append(list(time_df.loc[2]))
#     T_d.append(list(time_df.loc[3]))
#     T_p.append(list(time_df.loc[4]))
#     T_o.append(list(time_df.loc[5]))
#
# T_u = pd.DataFrame(T_u)
# T_a = pd.DataFrame(T_a)
# T_c = pd.DataFrame(T_c)
# T_d = pd.DataFrame(T_d)
# T_o = pd.DataFrame(T_o)
# T_p = pd.DataFrame(T_p)
#
# t_u = T_u.sum(axis=1).mean()
# t_a = T_a.sum(axis=1).mean()
# t_c = T_c.sum(axis=1).mean()
# t_d = T_d.sum(axis=1).mean()
# t_p = T_p.sum(axis=1).mean()
# t_o = T_o.sum(axis=1).mean()
#
# print(t_u, t_a, t_c, t_d, t_p, t_o)


# T_u = T_u.cumsum(axis=1)
# T_a = T_a.cumsum(axis=1)
# T_c = T_c.cumsum(axis=1)
# T_d = T_d.cumsum(axis=1)
# T_o = T_o.cumsum(axis=1)
# T_p = T_p.cumsum(axis=1)



# matplotlib.rcParams.update({'font.size': 20})
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
#
# quantile25_u = T_u.quantile(0.25, axis=0)
# quantile25_a = T_a.quantile(0.25, axis=0)
# quantile25_c = T_c.quantile(0.25, axis=0)
# quantile25_d = T_d.quantile(0.25, axis=0)
#
# quantile75_u = T_u.quantile(0.75, axis=0)
# quantile75_a = T_a.quantile(0.75, axis=0)
# quantile75_c = T_c.quantile(0.75, axis=0)
# quantile75_d = T_d.quantile(0.75, axis=0)
#
# average_u = T_u.mean(axis=0)
# average_a = T_a.mean(axis=0)
# average_c = T_c.mean(axis=0)
# average_d = T_d.mean(axis=0)
#
# plt.title(r'Runtime for OMD-AFW variants')
# plt.xlabel(r'Iterations')
# plt.ylabel(r'Running time (normzalied)')
#
# # plt.xscale('log')
# plt.yscale('log')
#
# # plt.ylim(ymin=5, ymax=1200)
# # plt.xlim(xmin=0, xmax=500)
#
# plt.fill_between(range(m), quantile25_u, quantile75_u, color='green', alpha=0.1)
# plt.fill_between(range(m), quantile25_a, quantile75_a, color='red', alpha=0.1)
# plt.fill_between(range(m), quantile25_c, quantile75_c, color='blue', alpha=0.1)
# plt.fill_between(range(m), quantile25_d, quantile75_d, color='orange', alpha=0.1)
#
# average_u_plot, = plt.plot(average_u, color='green', label='OMD-UAFW')
# average_a_plot, = plt.plot(average_a, color='red', label='OMD-ASAFW')
# average_c_plot, = plt.plot(average_c, color='blue', label='OMD-TSAFW')
# average_d_plot, = plt.plot(average_d, color='orange', label='OMD-A$^2$FW')
#
# plt.tight_layout()
#
# plt.legend(loc='lower right', fontsize=15)
# plt.savefig('runtime_4_shifted.png', dpi=300)
# plt.show()
#

# T_u, T_a, T_c, T_d, T_o = [], [], [], [], []
# runs = [20, 25, 50, 51, 52, 53, 54, 70, 71, 72, 73, 74, 75, 80, 81, 100, 101]
#
# for i in runs:
#     regret_df = pd.read_csv('regrets_4_' + str(i) + '.csv', index_col=0, header=None)
#     regret_df.index = range(6)
#     print(regret_df.sum(axis=1))
#     regret_df = regret_df.loc[[0, 1, 2, 3, 4, 5]]
#     regret_df = (1000 * regret_df)/regret_df.sum(axis=1)[0]
#
#     # print(i)
#     # print(regret_df.sum(axis=1))
#
#     T_u.append(list(regret_df.loc[0]))
#     T_a.append(list(regret_df.loc[1]))
#     T_c.append(list(regret_df.loc[2]))
#     T_d.append(list(regret_df.loc[3]))
#     T_o.append(list(regret_df.loc[5]))
#
# T_u = pd.DataFrame(T_u)
# T_a = pd.DataFrame(T_a)
# T_c = pd.DataFrame(T_c)
# T_d = pd.DataFrame(T_d)
# T_o = pd.DataFrame(T_o)
#
# T_u = T_u.cumsum(axis=1).to_numpy()
# T_a = T_a.cumsum(axis=1).to_numpy()
# T_c = T_c.cumsum(axis=1).to_numpy()
# T_d = T_d.cumsum(axis=1).to_numpy()
# T_o = T_o.cumsum(axis=1).to_numpy()
#
# for l in T_u:
#     print(l)
#     for i in range(len(l)):
#         l[i] = l[i]/(i + 1)
#
# for l in T_a:
#     for i in range(len(l)):
#         l[i] = l[i]/(i + 1)
#
# for l in T_c:
#     for i in range(len(l)):
#         l[i] = l[i]/(i + 1)
#
# for l in T_d:
#     for i in range(len(l)):
#         l[i] = l[i]/(i + 1)
#
# for l in T_o:
#     for i in range(len(l)):
#         l[i] = l[i]/(i + 1)
#
# T_u = pd.DataFrame(T_u)
# T_a = pd.DataFrame(T_a)
# T_c = pd.DataFrame(T_c)
# T_d = pd.DataFrame(T_d)
# T_o = pd.DataFrame(T_o)
#
#
# matplotlib.rcParams.update({'font.size': 20})
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
#
# quantile25_u = T_u.quantile(0.25, axis=0)
# quantile25_a = T_a.quantile(0.25, axis=0)
# quantile25_c = T_c.quantile(0.25, axis=0)
# quantile25_d = T_d.quantile(0.25, axis=0)
#
# quantile75_u = T_u.quantile(0.75, axis=0)
# quantile75_a = T_a.quantile(0.75, axis=0)
# quantile75_c = T_c.quantile(0.75, axis=0)
# quantile75_d = T_d.quantile(0.75, axis=0)
#
# average_u = T_u.mean(axis=0)
# average_a = T_a.mean(axis=0)
# average_c = T_c.mean(axis=0)
# average_d = T_d.mean(axis=0)
# average_o = T_o.mean(axis=0)
#
# plt.title(r'Regret for OMD-AFW variants')
# plt.xlabel(r'Iterations')
# plt.ylabel(r'Regret (normalized)')
#
# # plt.xscale('log')
# # plt.yscale('log')
#
# # plt.ylim(ymin=10, ymax=1200)
# # plt.xlim(xmin=0, xmax=500)
#
# plt.fill_between(range(m), quantile25_u, quantile75_u, color='green', alpha=0.1)
# plt.fill_between(range(m), quantile25_a, quantile75_a, color='red', alpha=0.1)
# plt.fill_between(range(m), quantile25_c, quantile75_c, color='blue', alpha=0.1)
# plt.fill_between(range(m), quantile25_d, quantile75_d, color='orange', alpha=0.1)
#
# average_u_plot, = plt.plot(average_u, color='green', label='OMD-UAFW')
# average_a_plot, = plt.plot(average_a, color='red', label='OMD-ASAFW')
# average_c_plot, = plt.plot(average_c, color='blue', label='OMD-TSAFW')
# average_d_plot, = plt.plot(average_d, color='orange', label='OMD-A$^2$FW')
# average_o_plot, = plt.plot(average_o, color='cyan', label='OFW')
#
# plt.tight_layout()
#
# plt.legend(loc='lower right')
# plt.savefig('regret_4_shifted.png', dpi=300)
# plt.show()
#
