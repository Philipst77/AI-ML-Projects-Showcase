# Write a Python function that computes the dot product of a matrix and a vector.
# The function should return a list representing the resulting vector if the operation 
# is valid, or -1 if the matrix and vector dimensions are incompatible. 
# A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns
# in the matrix equals the length of the vector. For example, an n x m matrix requires a vector 
# of length m.

import numpy as np
# Inital approach 
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    if len(a[0])!= len(b): #Checking if lenght of row in matrix is equal to length of vector
        #if not dot product does not work
        return -1 # Consequently return -1
    
    result =[]
    for row in a: # looping through rows in matrix a
        total =0 # intalize total variable
        for i in range(len(row)): # [0-n) inclusive 0 exclusive n 
            #for each number in this sequence from 0 up to one less then the length of the row
            # Where Doing for i in range(len(row)) because we need to perform operations that 
            # involve the elements at those positions
            total += row[i] * b[i]
        result.append(total)
    return result

            
        
#Second way of doing this is with Numpy Library which makes it really easy

def matrix_dot(a,b):
    
    a= np.array(a)
    b= np.array(b)
    #Before converstion a is just a list of list in order to repersent a matrix of rows
    # We convert both a and b to NumpyArrays 
    #which support fast vectorized math like a.dot(b) which runs in c which is way faster
    #Then python loops 
    # Also are consistent in type all numbers are stored in contigous memory not as seprate 
    # python objects
    # we basically convert them to numpy arrays so we can treat them as mathematical objects
    # not generic lists
    
    if a.shape[1] != b.shape[0]:
        return -1
    #.shape  returns a tuple (rows, cols)
    # so it just returns number of rows and cols in matrix and we do a.shape[1]
    # because the tuple is 0 index so in order to get col value from it and compare 
    #matrix a col value to vector b col val we need to get index 1 which corresponds to 
    # number of cols in matrix a because in order for us to do dotproduct of a and b 
    # the number of columns in each needs to be the same. 
    return a.dot(b).tolist()


z = np.array([[1,2,3],   # MATRIX OF 3X3 ROW,COL
              [4,5,6],
              [7,8,9]])

y = np.array([1,2,3]) # 1D ARRAY OF LEN 3 JUST VECTOR OF 3 ELEMENTS

x = np.array([[1,2,3,  #SINGLE ROW 9 COLUMNS
               4,5,6,
               7,8,9]])

print(z.shape)

print(y.shape)

print(x.shape)


# List Comperhension Approach

# just makes the code more compact

def matrix_dot_product(a,b):
    if len(a[0]) != len(b[0]):
        return -1
    
    return [sum(row[i] * b[i] for i in range(len(b))) for row in a]
    
#Numpy is a general purpose math library
#Pytorch is designed for deep learning and large scale tensor computation 

#Numpy Runs only on cpu
#Pytorch runs on GPU and specalized accelerators 

#Numpy use when your doing quick math, linear algebra and data preprocessing
#No need for gradients or GPU acceleration

#Pytorch  when you traing Machine Leanring Models or Deep Learning Models
# You need Gradients, GPUs, or accesses to deep learning eco system


#Pytorch Has autograd (Automatric  differntiation)

#Numpy if you want gradients(for training ML models) you have to code backprop by hand
#Pytorch: Tensors track operations and can compute gradients automatically with .backward()

import torch
x= torch.tensor(2.0, requires_grad=True)
Y = x**2 +3*x
y.backward()
print(x.grad)

