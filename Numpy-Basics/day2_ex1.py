import numpy as np

mt = np.array([[1,2,3], [4,5,6], [6,7,8]])
vc = np.array([9,0,10])
result = mt + vc
print("Add: ", result)

result_mul = mt * 2
print("Multiplication: \n ", result_mul)

dataset = np.random.randint(1,51, size=(5,5))
print("This is my random values: \n", dataset)

dataset[dataset > 25] = 0
print("Modified dataset: \n", dataset)

print("Sum: ", np.sum(dataset))
print("Mean: ", np.mean(dataset))
print("Standart deviation: ", np.std(dataset))
