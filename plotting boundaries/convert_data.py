import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def convert_data(matrix):

    row = matrix[0]
    for i in range(0,len(row)):
        if isinstance(row[i], str):
            if is_number(row[i]) == False:
                column = [row[i] for row in matrix] 
                listSet = list(set(column))

                # convert data from strings to int            
                for k in range(0,len(column)):
                    for j in range(0,len(listSet)):
                        if column[k] == listSet[j]:
                            column[k] = j

                    matrix[k][i] = column[k]


    # Convert to int
    matrix = np.array([[float(i) for i in row] for row in matrix])
    return matrix

