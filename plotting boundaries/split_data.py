import random
import numpy as np

'''
    Splits list matrix in two parts

    @param: matrix, a list matrix that should be split.
    @param: amount, a fraction of the data or the exact number of elements

    @return: matrix, split, two randomely distributed disjunct sets from the matrix.
'''
def split_data_list(matrix, amount):
    if amount < 1:
        split = random.sample(matrix, round(len(matrix)*amount))
    elif amount > 1:
        split = random.sample(matrix, int(amount))

    return [matrix, split]

