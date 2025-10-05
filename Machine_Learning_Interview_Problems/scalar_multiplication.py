# Actaul thinking solution thats needs a brain lmaooo

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    row = len(matrix) # number of rows in matrix 
    col = len(matrix[0]) # lenght of first row is the number of columns in teh matrix 
    result =[]
    for i in range(row):
        new_row =[]
        for j in range(col):
            new_row = matrix[i][j] * scalar
        result.append(new_row)
    return result



#Numpy solution
import numpy as np

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:

    arr = np.array(matrix)
    
    result = arr * scalar

    return result.tolist()