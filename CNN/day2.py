import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
import tensorflow as tf 

# Create a simulated image
image = np.random.rand(10, 10)

# print(image)

# Let's create a edge detector filter
edge_detector_kernel = np.array([
    [-1, -1, -1],
    [-1, -8, -1],
    [-1, -1, -1]    
])

blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9

# Apply the Convolution
edge_detected_image = convolve(image, edge_detector_kernel)
blurred_image = convolve(image, blur_kernel)

# Visualize original and filtered image
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Original Image')
# axes[1].imshow(edge_detected_image, cmap='gray')
# axes[1].set_title('Edge Image')
# axes[2].imshow(blurred_image, cmap='gray')
# axes[2].set_title('Blurred Image')
# plt.show()

#  Creating a simple input image tensor with batch size, height, width and channels 
im = tf.random.normal([1, 10, 10, 1])

# Defining a convolution layer
conv_layer = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same'
)

# Applying the convolution
output_tensor_image = conv_layer(im) # So what this does is take the image we created with tensor and passing to the conv_layer we created

print(f"Orinal image shape: {im.shape}")
print(f"Output shape: {output_tensor_image.shape}")