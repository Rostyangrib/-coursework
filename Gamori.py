from PIL.features import check

import SimplexGamori
import Simplex
import numpy as np
import math
import pulpi
from fractions import Fraction

def Print(answer):
    for j, i in enumerate(answer):
        print(f"x{j + 1} = {i}")

def GetFractional(arr):
    arr_frac = []
    for i in range(len(arr)):
        if math.copysign(1, arr[i]) == -1:
            x = math.ceil(abs(arr[i]))
            fractional_part = abs(-math.ceil(abs(arr[i])) - arr[i])
            arr_frac.append(round(fractional_part, 4))
        else:
            x = math.floor(arr[i])
            fractional_part = arr[i] - x
            arr_frac.append(round(fractional_part, 4))

    return arr_frac



def Gamori():
    array_val = []
    # c = np.array([2, 3, 1])
    # A = np.array([[9, 3, 4], [3, 4, 1], [1, 1, 1]])
    # b = np.array([5, 6, 7])

    # c = np.array([49000, 73000])
    # A = np.array([[22, 28], [10, 14], [1, 2], [2.2, 0.8], [1.2, 1.1]])
    # b = np.array([3200, 1500, 190, 210, 170])


    c = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    A = np.array([[2, -100, -400, -20, -200, -600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, -15, -200, -25, -50, -250, 0, 0, 0, 0, 0],
                  [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, -150, -200, -25, -350],

                  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])

    b = np.array([0, 0, 0, 5, 3, 40, 9, 2])



    answer, table = Simplex.simplex_method(c, A, b)
    arr_frac = np.array(GetFractional(answer))

    Print(answer)
    print(*arr_frac)

    max_frac = np.max(arr_frac)
    max_index = np.argmax(arr_frac)
    cnt = 0
    while max_frac != 0 or cnt > 20:
        m, n = table.shape
        cnt += 1
        #ищем строку в тайбле для составления нового ограничения
        ind_new_lim = 0
        for i in range(m):
            if int(table[i][0]) == max_index + 1:
                ind_new_lim = i
                break
        # получаем массив дробных частей найденной строки
        array_row_table_frac = np.array(GetFractional(table[ind_new_lim, 1:]))

        array_row_table_frac = array_row_table_frac * -1

        #готовим столбец и строку дял новой переменной
        array_row_table_frac = np.insert(array_row_table_frac, n-2, 1)
        array_row_table_frac = np.insert(array_row_table_frac, 0, 0)
        new_column = np.zeros((table.shape[0]))


        #вставляем
        table = np.insert(table, n - 1, new_column, axis=1)
        table = np.insert(table, m - 1, array_row_table_frac, axis=0)



        answer, table, answer_dict = SimplexGamori.simplex_method(table)
        m, n = table.shape
        base = []
        for i in range(m):
            base.append(int(table[i][0]))

        flags = []

        for i in range(m):
            if int(table[i][0]) - 1 < len(c) and table[i][n-1] % 1 == 0:
                flags.append(1)

        for i in range(len(c)):
            if not(i + 1 in base):
                flags.append(1)
        if  len(flags) == len(c):
            Print(answer)
            return
        flags.clear()
        arr_frac = np.array(GetFractional(answer))
        max_frac = np.max(arr_frac)
        max_index = np.argmax(arr_frac)
        array_val.append(float(table[0][n-1]))
        print(float(table[0][n-1]))
        if cnt > 50:
            break
    pulpi.build(array_val)
    average = sum(array_val) / len(array_val)
    print(average)

    number = 2.3

   # print(f" целевой функции {answer[0]}")


Gamori()

