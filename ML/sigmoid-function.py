import numpy as np  
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate the values
z = np.linspace(-15, 10, 100)
sigmoid_values = sigmoid(z)

# plot
plt.plot(z, sigmoid_values)
plt.title('Sigmoid Function')
plt.xlabel('Z')
plt.ylabel('o(z)')
plt.grid()
plt.show()
