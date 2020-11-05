import math
import numpy as np
from typing import List


def generate_p(n: int):
    """
    Generates matrix P
    """
    P = np.zeros((int(math.pow(2, n - 1)), n))
    for i in range(int(math.pow(2, n - 1))):
        for j in range(n):
            P[i][j] = (int(i/math.pow(2, j - 1)) % 2) * 2 - 1

    return P


def generate_a(n: int):
    P = generate_p(n)
    A = np.zeros((int(math.pow(2, n - 1)), int(n*(n - 1)/2)))
    # print(P)
    for i in range(int(math.pow(2, n - 1))):
        count = 0
        for j in range(n):
            for k in range(j):
                # print(i, j, k, count)
                # print(P[i][j], P[i][k])
                # print(A[i][count])
                A[i][count] = P[i][j] * P[i][k]
                count += 1

    return A

#
# def rank_of_P(n: int):
#     return np.linalg.matrix_rank(generate_p(n))

#
# for n in range(4, 5):
#     A = np.transpose(generate_a(n))
#     v1 = np.transpose([[0, 0, 0, 0, 0, 0, 0, 0]])
#     v2 = np.transpose([[0, 1/4, 1/4, 0, 1/4, 0, 0, 1/4]])
#     print(np.matmul(A, v1))
#     print(np.matmul(A, v2))
#     #print(n, np.linalg.matrix_rank(generate_a(n)))


def multiply(P, W):
    return np.matmul(np.matmul(np.transpose(P), W), P)


def diagonalize(w):
    k = len(w)
    W = np.zeros((k, k))
    for i in range(k):
        W[i][i] = w[i]

    return W


def generate_binary_strings(n: int):
    print('n = ', n)
    if n == 1:
        return [[-1], [1]]
    else:
        L = generate_binary_strings(n - 1)
        L0 = [[-1] + s for s in L]
        L1 = [[1] + s for s in L]
        print('n = ', n)
        return L0 + L1


def convert_binary_string_to_matrix(s, n):
    A = np.zeros((n, n), dtype=np.int64)
    count = 0
    for i in range(n):
        for j in range(n):
            A[i][j] = s[count]
            count += 1

    return A


b = 5
S = generate_binary_strings(b * b)
for s in S:
    A = convert_binary_string_to_matrix(s, b)
    det = abs(int(np.linalg.det(A)))
    if det != 0:
        print(det)

