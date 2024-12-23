from msilib.schema import tables
from tabnanny import check

import numpy as np
import pandas as pd


def FindFeta(table, m):
    row = table[m - 1]
    positive_values = 0
    max_index = -1
    max = -1
    for i, j in enumerate(row):
        if i > 2 and j is not None:
            if j - 0.05 > 0:
                positive_values += 1
            if j > max:
                max = j
                max_index = i

    return max_index, positive_values

def simplex_method(c, A, b):
    np.set_printoptions(linewidth=200)
    # добавить -с
    m, n = A.shape
    inf = 1000000
    # Расширенная матрица симплекс-таблицы (A | b)
    table = np.hstack([ b.reshape(-1, 1), A, -np.eye(m), np.eye(m)])
    new_cal = np.array([inf, inf, inf, 0])
    base_cal = np.array(["x6", "x7", "x8", None])
    new_row = np.zeros(table.shape[1])

    table = np.vstack([table, new_row])
    table = np.hstack([new_cal.reshape(-1, 1), base_cal.reshape(-1, 1), table])

    m, n = table.shape
    for i in range(2, n - 3):
        for j in range(0, m - 1):
            table[m - 1][i] += table[j][i] * table[j][0]


    max_index, positive_values = FindFeta(table, m)
    while positive_values > 0:

        feta = np.zeros(m - 1)
        for i in range(m - 1):
            feta[i] = table[i][2] / table[i][max_index]
            if feta[i] <= 0:
                feta[i] = inf
        min_index = np.argmin(feta)

        table[min_index, 2:] = table[min_index, 2:] / table[min_index][max_index]

        for i in range(m):
            if i != min_index:
                check = table[i][max_index]
                for j in range(2, n):
                    table[i][j] = table[i][j] - (check * table[min_index][j])

        df = pd.DataFrame(table)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', None)
        print(df)
        print(feta)

        max_index, positive_values = FindFeta(table, m)

    df = pd.DataFrame(table)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)

    print(df)




# Пример данных
# c = np.array([12, 16])  # Целевая функция (-3x1 - 2x2)
# A = np.array([[2, 6], [5, 4], [2, 3]])  # Ограничения: 2x1 + x2 ≤ 8, x1 + 2x2 ≤ 6
# b = np.array([24, 31, 18])

c = np.array([12, 16])  # Целевая функция (-3x1 - 2x2)
A = np.array([[8, 10], [5, 4], [2, 3]])  # Ограничения: 2x1 + x2 ≤ 8, x1 + 2x2 ≤ 6
b = np.array([10, 31, 18])

# Решаем задачу
optimal_solution = simplex_method(c, A, b)
print(f"Оптимальное решение: {optimal_solution}")
