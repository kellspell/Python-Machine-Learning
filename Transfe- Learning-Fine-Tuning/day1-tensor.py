import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Display model architecture
# model.summary()

# access specific layers
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")

   