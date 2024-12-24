import math

import numpy as np
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

def FindFeta(table, m):
    row = table[m - 1]
    neg_values = 0
    max_abs_index = -1
    max = 1000000
    for i, j in enumerate(row):
        if i > 0 and j is not None:
            if j < 0:
                neg_values += 1
            if j < max:
                max = j
                max_abs_index = i

    return max_abs_index, neg_values

def CalculateDelta(table, padded):
    delta = []
    n, m = table.shape
    dl = 0
    for i in range(1, m):
        dl = 0
        for j in range(n - 1):
            a = padded[int(table[j][0])-1]
            b = table[j][i]
            dl += (padded[int(table[j][0])-1]*table[j][i])
        dl = dl - padded[i - 1]
        delta.append(dl)
    delta.insert(0, 0)
    return delta

def MinIndex(feta):
    min_index = 0
    min = 10000
    for k, h in enumerate(feta):
        if math.copysign(1, h) == 1 and h != 1000000 and h < min:
            min = h
            min_index = k

    return min_index, min

def MatrixApdate(table, min_index, max_index):
    m, n = table.shape
    table[min_index][0] = max_index
    table[min_index, 1:] = table[min_index, 1:] / table[min_index][max_index]
    table = np.round(table, 2)
    for i in range(m - 1):
        if i != min_index:
            check = table[i][max_index]
            for j in range(1, n):
                table[i][j] = table[i][j] - (check * table[min_index][j])

    return table

def simplex_method(c, A, b):
    np.set_printoptions(linewidth=200)

    m, n = A.shape
    inf = 1000000

    table = np.hstack([A, np.eye(m), b.reshape(-1, 1)])

    base_cal = np.array([17, 18, 19, 20, 21, 22, 23, 24, 0])

    padded_arr = np.pad(c, (0, table.shape[1] - len(c)), mode='constant', constant_values=0)

    table = np.vstack([table, padded_arr])
    table = np.hstack([base_cal.reshape(-1, 1), table])
    m, n = table.shape

    delta = CalculateDelta(table, padded_arr)

    table[m - 1] = delta
    max_index, neg_values = FindFeta(table, m)
    while neg_values > 0:

        feta = np.zeros(m - 1)
        for i in range(m - 1):
            if table[i][max_index] != 0:
                feta[i] = float(table[i][n - 1]) / float(table[i][max_index])
            else:
                feta[i] = inf
                continue
            if feta[i] < 0:
                feta[i] = inf

        min_index, min = MinIndex(feta)

        table = MatrixApdate(table, min_index, max_index)

        df = pd.DataFrame(table)

        print(df)
        delta = CalculateDelta(table, padded_arr)
        table[m - 1] = delta
        max_index, neg_values = FindFeta(table, m)

    df = pd.DataFrame(table)
    print(df)



# c = np.array([12, 16])
# A = np.array([[2, 6], [5, 4], [2, 3]])
# b = np.array([24, 31, 18])

# c = np.array([4, 5, 6])
# A = np.array([[1, 2, 3], [4, 3, 2], [3, 1, 1]])
# b = np.array([35, 45, 40])

c = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
A = np.array([[2, -100, -400, -20, -200, -600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, -15, -200, -25, -50, -250, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, -150, -200, -25, -350],

              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])

b = np.array([0, 0, 0, 5, 3, 40, 9, 2])

optimal_solution = simplex_method(c, A, b)
print(f"Оптимальное решение: {optimal_solution}")
