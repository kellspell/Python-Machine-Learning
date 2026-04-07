import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, maximum_filter, uniform_filter
import tensorflow as tf 

# Create  a sample feature
feature_imape = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [0, 1, 2, 3]
    
])

# Max Pooling (2x2)
max_pooled = maximum_filter(feature_imape, size=2, mode='constant')

# Average pooling (2x2)
avg_pooled = uniform_filter(feature_imape, size=2, mode='constant')

# Visualize original and filtered image
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(feature_imape, cmap='viridis')
# axes[0].set_title('Original Feature image')
# axes[1].imshow(max_pooled, cmap='viridis')
# axes[1].set_title('Max Pooled Image')
# axes[2].imshow(avg_pooled, cmap='viridis')
# axes[2].set_title('Average Pooled Image')
# plt.show()

# Create a sample image input in tensorflow (1x4x4x1) represents: batch_size, height, width, channels 
im_tensor = tf.constant(feature_imape.reshape(1, 4, 4, 1), dtype=tf.float32)

# Create a Pooling layer
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
max_pooled_tensor = max_pool(im_tensor) 

# the Average now
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')
avg_pooled_tensor = avg_pool(im_tensor) 

print(f"Max Pooled Tensor:\n{tf.squeeze(max_pooled_tensor).numpy()}")
print(f"AVG Pooled Tensor:\n{tf.squeeze(avg_pooled_tensor).numpy()}")
