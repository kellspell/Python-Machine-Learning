import numpy as np  

# Creating Arrays
arr = np.array([1,2,3,4,5])
# print(arr)

# built in functions
zeros = np.zeros([3,3])
# print(zeros)

ones = np.ones([2,4])
# print(ones)

range_array = np.arange(1, 10, 2)
# print(range_array)

linspace_array = np.linspace(0, 1, 5, 10)
# print(linspace_array)

# Reshapping array 
# arr = np.array([1,2,3,4,5,6])
# reshaped = arr.reshape((2,3))
# print(reshaped)

# Expand Arrays
# arr = np.array([1,2,3,4,5,6])
# expanded = arr[:, np.newaxis]
# print(expanded)

# Maths with numpy
# a = np.array([1,24,6])
# b = np.array([9,94,78])

# print(a + b)
# print(a - b)
# print(a * b)
# print(a / b)
# print(a % b)

# arr = np.array([4,5,6])
# print(np.sqrt(arr))
# print(np.sum(arr))
# print(np.mean(arr))
# print(np.max(arr))
# print(np.min(arr))

# Indexing 
# arr = np.array([40,50,60, 70, 80])
# print(arr[2])
# print(arr[-1])
# print(arr[1:3])
# print(arr[:4])

# Tanspose 
matrix = np.array([[1,2,3], [3,4,5], [6,7,8]])
print("Original Matrix: \n",matrix)

transpose = matrix.T
print("Transpose: \n", transpose)