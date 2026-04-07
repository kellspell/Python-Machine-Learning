import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the dataset
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Apply one-hot encode
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True  
)

# Fit the generator to training data
datagen.fit(X_train)

# Creating a model
def create_model():
    model = models.Sequential()
    
    # Convolutional layer 1
    model.add(layers.Input(shape=(32, 32, 3)))  # Fixed: added 'shape=' and parentheses
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))  # Fixed: changed 'MaxPoolinf2D' to 'MaxPooling2D'
    model.add(layers.Dropout(0.25))  # Fixed: changed 2.25 to 0.25 (dropout should be between 0-1)
    
    # Convolutional layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))  # Fixed: changed 'MaxPoolinf2D' to 'MaxPooling2D'
    model.add(layers.Dropout(0.25))  # Fixed: changed 2.25 to 0.25
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))  # Fixed: changed 2.25 to 0.25
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Let's compile the model (FIXED: moved outside the function)
model = create_model()
model.compile(
    optimizer='adam',  # Fixed: changed 'Adan' to 'adam' (Adan isn't a standard optimizer in Keras)
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model using the augmented data generator 
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(X_test, y_test),
    steps_per_epoch=X_train.shape[0] // 64
)  

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")   

# Visualization
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Num of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.xlabel('Num of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()