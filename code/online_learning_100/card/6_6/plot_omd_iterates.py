import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from plot_constants import *
# Make sure to see the plot_constants file if you want to make sense of what is happeneing here

k = 4
variant_indices = [0, 1, 2, 3]
for i in runs:      # For each run of the experiment
    # Extract times
    iterates_df = pd.read_csv('iterates_' + filename_prefix + str(i) + '.csv', index_col=0,
                           header=None)
    iterates_df.index = range(k)                    # Reindex dataframe
    # Extract values only for the required variants
    iterates_df = iterates_df.loc[variant_indices]
    # Normalize UAFW values to normalized_value = 1000
    iterates_df = (normalized_value * iterates_df) / iterates_df.sum(axis=1)[0]
    for j in variant_indices:                          # Save these times to T for each variant
        T[j].append(list(iterates_df.loc[j]))

for j in variant_indices:                          # For each variant
    T[j] = pd.DataFrame(T[j])               # Convert to pandas dataframe; easier to use than array
    T[j] = T[j].cumsum(axis=1)              # Convert T to have cumulative times over the iterations

    # Extract upper quantile values for each iteration across runs
    quantile_lower[j] = T[j].quantile(lower_quantile_value, axis=0)
    # Extract lower quantile values for each iteration across runs
    quantile_upper[j] = T[j].quantile(upper_quantile_value, axis=0)
    # Take average across runs
    average[j] = T[j].mean(axis=0)

for j in variant_indices:
    print(average[j][m - 1])


matplotlib.rcParams.update({'font.size': 20})       # Change matplotlib font size
plt.rc('text', usetex=True)                         # Use LaTeX
plt.rc('font', family='serif')                      # Change font family to serif

plt.title(r'Number of AFW iterations for OMD-AFW variants')          # Plot title
# x-axis corresponds to outer iterations - i.e. number of points projected so far
plt.xlabel(r'Outer loop iterations')
# y-axix corresponds to cumulative value of the quantity being plotted
plt.ylabel(r'AFW iterations (normzalied)')

# plt.xscale('log')
# plt.yscale('log')

# Uncomment to adjust axes ranges
# plt.ylim(ymin=5, ymax=1200)
# plt.xlim(xmin=0, xmax=500)

for j in variant_indices:      # For each variant
    # Fill with solid line the average value across runs of the quantity being plotted
    average_plots[j], = plt.plot(average[j], color=colors[j], label=labels[j])
    # Fill with transparent color in the quantile range for the quantity being plotted
    plt.fill_between(range(m), quantile_lower[j], quantile_upper[j], color=colors[j], alpha=0.1)

plt.tight_layout()

plt.legend(loc='upper left', fontsize=15)      # Location and size of legend
plt.savefig('iterations.png', dpi=300)          # Figure name
# plt.show()                                      # Show plot
