import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	arr = np.array(a)

	if arr.size != new_shape[0] * new_shape[1]:
		return []

	reshaped_matrix= arr.reshape(new_shape)

	return reshaped_matrix.tolist()



# No numpy Library calls


def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    
   row, col = new_shape
   
   flat = [num for row in a for num in row ]
    #This above is equivalent to 
    #    flat=[]
    #    for row in a:
    #        for num in row:
    #            flat.append(num)
    # The thing above is just a list comperhension its compact form of this 

   if len(flat) != row * col:
            return []

        # Fill new matrix row by row
   result = []
   index = 0
   for i in range(row):
       new_row = flat[index:index + col]
       result.append(new_row)
       index += col
       
       return result