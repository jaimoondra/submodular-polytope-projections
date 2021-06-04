from utils import *
import itertools


def pool(values, weights, l, r, ):
    new_point = sum(map(lambda x: values[x] * weights[x], range(l, r + 1))) / sum(weights[l: r + 1])
    values[l] = new_point
    weights[l] = sum(weights[l: r + 1])

    return values[:l + 1], weights[:l + 1]


def poolAdjacentViolators(input):
    """
    Main function to solve the pool adjacent violator algorithm
    on the given array of data.
    This is a O(n) implementation. Trick is that while regersssing
    if we see a violation, we average the numbers and instead of
    storing them as two numbers, we store the number once and store
    a corresponding weight. This way, for new numbers we don't have
    to go back for each n, but only one place behind and update the
    corresponding weights.
    """
    weights = []
    output = []

    index = 0
    while index < len(input):
        temp = index

        # Find monotonicity violating sequence, if any.
        # Difference of temp-beg would be our violating range.
        while temp < len(input) - 1 and input[temp] > input[temp + 1]:
            # Append this number to the final output and set its weight to be 1.
            temp += 1

        if temp == index:
            output_beg = len(output) - 1
            output_end = output_beg + 1
            output.append(input[index])
            weights.append(1)
            index += 1
        else:
            # Pool the violating sequence, if after violating monotonicity
            # is broken, we need to fix the output array.
            output_beg = len(output)
            output_end = output_beg + temp - index
            output.extend(input[index: temp + 1])
            weights.extend([1] * (temp - index + 1))
            index = temp + 1

        # Fix the output to be in the increasing order.
        while output_beg >= 0 and output[output_beg] > output[output_beg + 1]:
            output, weights = pool(output, weights, output_beg, output_end)
            diff = (output_end - output_beg)
            output_beg -= 1
            output_end -= diff

    return np.array(
        list(itertools.chain(*map(lambda i: [output[i]] * weights[i], range(len(weights))))))


def isotonic_projection(y, g):
    """
    Main function to compute a projection over cardinality based B(f).
    This functions solved the dual problem in the dual space using the
    PAV algorithm and then maps it back to the primal. The inputs are
    the point y we are trying to project and a dictionary g with the
    submodular function. g is of the form {0:0, 1:f(1),..., n:f(n)}
    """

    n = len(g) - 1
    pi = np.argsort(-y)
    C = np.array([g[i] - g[i - 1] for i in range(1, n + 1)])
    C_ = {}
    for i, j in enumerate(pi):
        C_[j] = C[i]

    C_ = np.array([C_[i] for i in sorted(C_.keys())])

    error = C_ - y
    error_sorted = error[pi]
    z = poolAdjacentViolators(error_sorted)

    z_ = {}
    for i, j in enumerate(pi):
        z_[j] = z[i]

    return np.array([z_[i] for i in sorted(z_.keys())]) + y
