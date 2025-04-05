import math

import numpy as np
import pandas as pd
import mpmath

from fractions import Fraction

# Настройка отображения для numpy
np.set_printoptions(precision=3)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

def FindFeta(table, m):
    row = table[m - 1]
    neg_values = 0
    max_abs_index = -1
    max = 1000000
    for i, j in enumerate(row):
        if i > 0 and j is not None:
            if math.copysign(1, j) == -1:
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

    for i in range(m):
        if i != min_index:
            check = table[i][max_index]
            for j in range(1, n):
                table[i][j] = Fraction(table[i][j] - (check * table[min_index][j])).limit_denominator(1000)

    return table

def simplex_method(c, A, b):
    np.set_printoptions(linewidth=200)

    m, n = A.shape
    inf = 1000000

    table = np.hstack([A, np.eye(m), b.reshape(-1, 1)], dtype=object)
    #тут проблемка mm
    cnt = n
    base_cal = np.array([])
    while cnt < n + m:
        base_cal = np.append(base_cal, cnt + 1)
        cnt += 1

    base_cal = np.append(base_cal, 0)

    padded_arr = np.pad(c, (0, table.shape[1] - len(c)), mode='constant', constant_values=0)

    table = np.vstack([table, padded_arr])
    table = np.hstack([base_cal.reshape(-1, 1), table])
    m, n = table.shape

    for i in range(m):
        for j in range(1, n):
            table[i][j] = Fraction(table[i][j]).limit_denominator(100)

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

        #print(df)
        delta = CalculateDelta(table, padded_arr)
        table[m - 1] = delta
        max_index, neg_values = FindFeta(table, m)

    answer = [0] * (n - 1)
    for i in range(m - 1):
        answer[int(table[i][0]) - 1] = table[i][n - 1]

    df = pd.DataFrame(table)
    #print(df)
   # print(*answer)
    return answer, table


# c = np.array([12, 16])
# A = np.array([[2, 6], [5, 4], [2, 3]])
# b = np.array([24, 31, 18])

# пример для демо метода гамори
c = np.array([2,3 , 1])
A = np.array([[9, 3, 4], [3, 4, 1], [1, 1, 1]])
b = np.array([5, 6, 7])

# c = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# A = np.array([[2, -100, -400, -20, -200, -600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0, -15, -200, -25, -50, -250, 0, 0, 0, 0, 0],
#               [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, -150, -200, -25, -350],
#
#               [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
#
# b = np.array([0, 0, 0, 5, 3, 40, 9, 2])

#optimal_solution = simplex_method(c, A, b)
print("Оптимальное решение:")
