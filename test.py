import numpy as np

# create two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# matrix multiplication using numpy.dot()
C = np.dot(A, B)

# matrix multiplication using @ operator
D = A @ B

# matrix multiplication using np.matmul()
E = np.matmul(A, B)

# matrix multiplication using * operator
F = A * B

# print the result
print("C = \n", C)
print("D = \n", D)
print("E = \n", E)
print("F = \n", F)