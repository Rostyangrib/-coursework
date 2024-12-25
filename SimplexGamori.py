import math

import numpy as np
import pandas as pd


# Настройка отображения для numpy
np.set_printoptions(precision=3)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

def FindFeta(row):
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
    table = np.round(table, 2)
    for i in range(m - 1):
        if i != min_index:
            check = table[i][max_index]
            for j in range(1, n):
                a = table[i][j]
                table[i][j] = round(round(table[i][j], 4) - round(check * table[min_index][j], 4), 4)
                a = table[i][j]
                if round(abs(table[i][j]), 3) <= 0.005:
                    table[i][j] = 0

    table = np.round(table, 4)
    return table

def simplex_method(table):
    np.set_printoptions(linewidth=200)
    inf = 1000000

    m, n = table.shape


    ch = table[:-1, n-1]
    idx = -1
    has_negatives = np.any(table[:, n-1] < 0)
    while has_negatives:
        for i, j in enumerate(ch):
            if math.copysign(1, j) == -1:
                idx = i
                break
        if idx != -1:
            #???
            min_index, neg_values = FindFeta(table[idx, 1:-1])
            table = MatrixApdate(table, idx, min_index + 1)

        has_negatives = np.any(table[:, n-1] < 0)
        idx = -1
        ch = table[:-1, n - 1]

    max_index, neg_values = FindFeta(table[m-1])

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

    answer = [0] * (n - 1)
    for i in range(m - 1):
        answer[int(table[i][0]) - 1] = table[i][n - 1]

    df = pd.DataFrame(table)
    print(df)
   # print(*answer)
    return answer, table

