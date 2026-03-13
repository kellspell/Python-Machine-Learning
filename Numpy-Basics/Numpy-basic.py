# Today We'll learn about numpy broadcasting
import numpy as np

# Array and scalar broadcasting
# arr = np.array([1,2,3,4,5,6])
# print(arr + 10)

# mt = np.array([[1,2,3],[1,7,3]])
# vc = np.array([[1,9,8]])
# print(mt + vc)

# aggregation functions
mt = np.array([[1,2,3],[1,7,3]])
# print("Sum: ", np.sum(mt))
# print("Mean: ", np.mean(mt))
# print("Max: ", np.max(mt))
# print("Min: ", np.min(mt))
# print("Standart deviation: ", np.std(mt))
# print("Sum along rows: ", np.sum(mt, axis=1))
# print("Sum along columns: ", np.sum(mt, axis=0))

# boolean indexing and filtering 
arr = np.array([1,2,3,4,5,6])
evens = arr[arr % 2 == 0]
print("Evens: ", evens)

arr[arr > 3] = 0
print('Modified Array: ', arr)

# We always get differents array number because we dont have set the Seed function
# This ensure us reproducibility 
np.random.seed(42)

# Random numbers generation and setting seeds
rand_arr = np.random.rand(3,3)
print("Random Array: \n", rand_arr)

rand_int = np.random.randint(0, 10, size=(3,3))
print("Random integers array: \n", rand_int)




