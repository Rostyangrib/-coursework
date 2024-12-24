from msilib.schema import tables
from tabnanny import check

import numpy as np
import pandas as pd
from numpy.ma.core import shape


def FindFeta(table, m):
    row = table[m - 1]
    neg_values = 0
    max_abs_index = -1
    max = 1000000
    for i, j in enumerate(row):
        if i > 0 and j is not None:
            if float(j) < 0:
                neg_values += 1
            if float(j) < float(max):
                max = j
                max_abs_index = i

    return max_abs_index, neg_values


def simplex_method(c, A, b):
    np.set_printoptions(linewidth=200)
    # добавить -с
    for i in range(len(c)):
        c[i] *= -1
    m, n = A.shape
    inf = 1000000

    table = np.hstack([A, np.eye(m), b.reshape(-1, 1)])
    # new_cal = np.array([inf, inf, inf, 0])
    base_cal = np.array([17, 18, 19, 20, 21, 22, 23, 24, 0])
    new_row = np.zeros(table.shape[1])
    padded_arr = np.pad(c, (0, table.shape[1] - len(c)), mode='constant', constant_values=0)

    table = np.vstack([table, padded_arr])
    table = np.hstack([base_cal.reshape(-1, 1), table])
    m, n = table.shape

    max_index, neg_values = FindFeta(table, m)
    while neg_values > 0:

        feta = np.zeros(m - 1)
        for i in range(m - 1):
            if table[i][max_index] != 0:
                feta[i] = float(table[i][n - 1]) / float(table[i][max_index])
            else:
                feta[i] = inf
            if feta[i] <= 0:
                feta[i] = inf
        min_index = np.argmin(feta)
        table[min_index][0] = max_index
        table[min_index, 1:] = table[min_index, 1:] / table[min_index][max_index]
        table = np.round(table, 2)
        for i in range(m):
            if i != min_index:
                check = table[i][max_index]
                for j in range(1, n):
                    table[i][j] = table[i][j] - (check * table[min_index][j])

        df = pd.DataFrame(table)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', None)
        print(df)
        print(feta)

        max_index, neg_values = FindFeta(table, m)

    df = pd.DataFrame(table)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)

    print(df)


# Пример данных
# c = np.array([12, 16])  # Целевая функция (-3x1 - 2x2)
# A = np.array([[2, 6], [5, 4], [2, 3]])  # Ограничения: 2x1 + x2 ≤ 8, x1 + 2x2 ≤ 6
# b = np.array([24, 31, 18])

# c = np.array([4, 5, 6])  # Целевая функция (-3x1 - 2x2)
# A = np.array([[1, 2, 3], [4, 3, 2], [3, 1, 1]])  # Ограничения: 2x1 + x2 ≤ 8, x1 + 2x2 ≤ 6
# b = np.array([35, 45, 40])

c = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Целевая функция (-3x1 - 2x2)
A = np.array([[2, -100, -400, -20, -200, -600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, -15, -200, -25, -50, -250, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, -150, -200, -25, -350],

              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])

b = np.array([0, 0, 0, 5, 3, 40, 9, 2])

# Решаем задачу
optimal_solution = simplex_method(c, A, b)
print(f"Оптимальное решение: {optimal_solution}")
