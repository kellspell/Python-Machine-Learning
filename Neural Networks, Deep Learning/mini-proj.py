import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the pixel value to [0.1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Apply One-Hot encode for out labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# print out data shape
print(f"Training Data: {X_train.shape}, {y_train.shape}")
print(f"Testing Data: {X_test.shape}, {y_test.shape}")


# Define the model baseline
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax') 
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model baseline
history = model.fit(
    X_train, y_train,
    validation_split = 0.2,
    epochs = 10,
    batch_size = 64,
    verbose = 1
)

# Evaluate the model baseline
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Baseline model accuracy: {accuracy:.4f}")

# Improved model
improved_model = Sequential([
    Conv2D(64, (5, 5), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax') 
])

# Compiled the improved model with learning rate scheduler
optmizer = tf.keras.optimizers.Adam(learning_rate=0.001)
improved_model.compile(optimizer=optmizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Improved_model 
improved_history = improved_model.fit(
    X_train, y_train,
    validation_split = 0.2,
    epochs = 20,
    batch_size = 64,
    verbose = 1
)

# Evaluate the improved_model 
improved_loss, improved_accuracy = improved_model.evaluate(X_test, y_test, verbose=0)
print(f"Improved model accuracy: {improved_accuracy:.4f}") 

# Visualize the improved model
plt.plot(improved_history.history['accuracy'], label='Training accuracy')
plt.plot(improved_history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize the Loss Function
plt.plot(improved_history.history['loss'], label='Training loss')
plt.plot(improved_history.history['val_loss'], label='Validation loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()