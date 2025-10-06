import numpy as np 
#Numpy Solution
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    arrayA_row = len(a)
    arrayA_col = len(a[0])
    arrayB_row = len(b)
    arrayB_col = len(b[0])

    if arrayA_col != arrayB_row:
        return -1

    arraya = np.array(a)
    arrayb = np.array(b)
   
    result = arraya.dot(arrayb)
    return result



def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    
    
    rowa = len(a)
    rowb = len(b)
    cola = len(a[0])
    colb = len(b[0])
    
    if cola != rowb:
        return -1
    
    
    result =[]
    for i in range(rowa):
        new_row=[]
        for j in range(colb):
            val =0
            for k in range(cola):
                val += a[i][j] * b[k][j]
                new_row.append(val)
            result.append(new_row)
    
    return result

