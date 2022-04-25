n, m = 100, 2000            # n: dimension, m: number of projected points (outer iterations)
k = 6                       # Total number of variants
variant_indices = [0, 1, 2, 3, 4, 5]
# Each index in variant_indices corresponds to an optimization method for online problems. Here is
# the mapping of to method:
# 0: OMD-UAFW       OMD with unoptimized/vanilla Away-Step Frank-Wolfe (AFW)
# 1: OMD-ASAFW      OMD with active set optimized AFW
# 2: OMD-TSAFW      OMD with tight set optimized AFW
# 3: OMD-A2FW       OMD with adaptive AFW (A2FW)
# 4: OFW            Online Frank-Wolfe
# 5: OMD-PAV        OMD with isotonic regression/pool-adjacent violators algorithm (for
# cardinality-based polytopes only)
# Therefore, if one wishes to plot some quantity (say runtimes) only for OMD-UAFW, OFW,
# and OMD-PAV, one needs to change variant_indices to [0, 4, 5]

alpha = 0.1                 # opacity value for quantile plots
lower_quantile_value, upper_quantile_value = 0.10, 0.90         # quantile range
# Indices of independent runs of the experiment. Each index in the list corresponds to a
# different run of the experiment
runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
a, b = 1, 2         # Values of a, b
# Used to describe how the data file is named
filename_prefix = str(n) + '_' + str(m) + '_' + str(a) + '_' + str(b) + '_'
# Colors for different variants. The variants, in order, are: OMD-UAFW, OMD-ASAFW, OMD-TSAFW,
# OMD-A2FW, OFW, OMD-PAV
colors = ['green', 'red', 'blue', 'orange', 'cyan', 'magenta']
labels = ['OMD-UAFW', 'OMD-ASAFW', 'OMD-TSAFW', 'OMD-A$^2$FW', 'OFW', 'OMD-PAV']
# This is the value all data for OFW-UAFW will be normalized to
normalized_value = 1000

# List for all data. T[0] contains data for OMD-UAFW for instance
T = []
for j in range(k):
    T.append([])

# List for lower quantile values
quantile_lower = [[]] * k

# List for upper quantile values
quantile_upper = [[]] * k

# List for average values
average = [[]] * k

# List for average value plots. Will store matplotlib objects
average_plots = [[]] * k
