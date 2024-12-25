import SimplexGamori
import Simplex
import numpy as np
import math

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
    c = np.array([2, 3, 1])
    A = np.array([[9, 3, 4], [3, 4, 1], [1, 1, 1]])
    b = np.array([5, 6, 7])



    answer, table = Simplex.simplex_method(c, A, b)
    arr_frac = np.array(GetFractional(answer))

    Print(answer)
    print(*arr_frac)

    max_frac = np.max(arr_frac)
    max_index = np.argmax(arr_frac)

    while max_frac != 0:
        m, n = table.shape

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

        answer, table = SimplexGamori.simplex_method(table)
        arr_frac = np.array(GetFractional(answer))
        max_frac = np.max(arr_frac)
        max_index = np.argmax(arr_frac)
        print(max_frac)

    number = 2.3

   # print(f" целевой функции {answer[0]}")


Gamori()

