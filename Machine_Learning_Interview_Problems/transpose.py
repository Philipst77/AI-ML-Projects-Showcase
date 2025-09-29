# import numpy as np

# def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
# 	return np.transpose(a)





a=[[1,2,3],[4,5,6]]


def transpose_matrix(a):
    
    rows = len(a)
    cols = len(a[0])
    
    # when you write i that means you care about the variable your going to use it later
    # when you use _ its a Throwaway Variable  you saing i need to repeating something row
    # times  but i dotn care about the loop variable 
    
    result = [[0 for _ in range((rows))] for _ in range((cols))]
    
    # We dont do range(len(rows)) or cols because they are already integers
    #doing len rows will error out since there not a list
    
    
    for i in range(rows):
        for j in range(cols):
            result[j][i] = a[i][j]
        
    return result 


print(transpose_matrix(a))