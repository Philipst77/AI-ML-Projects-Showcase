import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:

    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    det = a*d - b*c
    if det !=0:
        inv = [
            [d/det, -b/det],
             [-c/det , a/det]
        ] 
    
    else:
        return None

    return inv


#Numpy Version

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:

    arr = np.array(matrix)
    det = np.linalg.det(arr)
    
    if det != 0:
        inv = np.linalg.inv(arr)
        return inv.tolist()
    
    return None