from tensorflow.keras.datasets import mnist, cifar10
import tensorflow as tf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Load the mnist dataset
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
print(f" Mnist Dataset: Train - {X_train_mnist.shape}")

# Load our second dataset
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
print(f" Cifar10 Dataset: Train - {X_train_cifar.shape}")

# Difine basic Dense Layer
layer = tf.keras.layers.Dense(units=10, activation='relu')
print(f"Tensoflow layers: {layer}")

# Difine basic pytorch Dense Layer
layer = nn.Linear(in_features=10, out_features=5)
print(f"Pytorch layers: {layer}")

# Visualize the dataset
plt.imshow(X_test_mnist[0], cmap='gray')
plt.title(f'Mnist Label {y_test_mnist[0]}')
plt.show()


plt.imshow(X_test_cifar[0], cmap='gray')
plt.title(f'Cifar-10 Label {y_test_cifar[0]}')
plt.show()

