import numpy as np  

a = np.array([[1,2], [2,1]])
b = np.array([[4,5], [3,9]])

c = 2 * a
# print("Scalar Multiplication: \n", c)

res = np.dot(a,b)
# print("Matrix multiplication: \n", res)

I = np.eye(5)
# print("Identity Matrix: \n", I)

Z = np.zeros((2,3))
# print(Z)

D = np.diag([1,2,3,4,5,6])
print("Diagonal Matrix: \n", D)

"""
Determinants and inverse of a matrix
* Determinants 
    * Scalar value that provides informations about a matrix properties
    * Only for square matrices
    * det(A) = 0, the matrix A is singular
    * det(A) != 0, is invetible
    * Geometric interpretation
        * For 2 x 2 matrix, The determinant represents the Scaling factor area
        formed by  its column vector
* Inverse of Matrix
    * Denoted as A**1
    * Product of a Matrix and its inverse is the identity matrix A x A
    * Matrix is invertible only if the det(A) != 0
    * Formula for 2X2 2x2        
"""

A = np.array([[2,3],[5, 8]])
determinant = np.linalg.det(A)
print("Dterminant: \n", determinant)

inverse = np.linalg.inv(A)
print("Inverse: \n", inverse)

"""
* Eigenvalues & Eigenvectors
    * Eigenvectors point in the directions where the matrix transformation streches or compressses vectors
    * Eigenvalues indicates the factor of streches or compression
        * Properties 
            * Matrix of size N x N has N eigenvalues and eigenvectors
            * Eigenvalues can be real or complex
            * For a symmetric matrix, eigenvalues are alway real 
"""

eigenValues, eigenVectors = np.linalg.eig(A)
print("EigenVal: \n", eigenValues)
print("EigenVec: \n", eigenVectors)


"""
* Matrix Dicomposition
    * Process of breaking a matrix into simpler components to analyze or solve problems 
    * Singular value decomposition(SVD)
        * SVD decomposes a matrix A into 3 matrices 
"""

U, S, Vt = np.linalg.svd(A)
print("U: \n", U)
print("S Singluar value: \n", S)
print("V Transpose: \n", Vt)

