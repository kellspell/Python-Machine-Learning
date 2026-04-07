import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import tensorflow as tf
import torch.nn as nn

# Transforming images to tensors 
im_transform = transforms.ToTensor() # this line here gets the image and transforms into Tensors

# Loading the dataset
train_dataset = datasets.CIFAR10(root='./home/Kellspell/Py-Dev/CNN', train=True, transform=im_transform, download=True)  

# Visualize sample data
fig, axes = plt.subplots(1, 5, figsize=(12, 3))  
for i in range(5): # What we are doing here is to loop through the first 5 images on the array 
    image, label = train_dataset[i] 
    axes[i].imshow(image.permute(1, 2, 0))
    axes[i].axis('off')
    axes[i].set_title(f'Label: {label}')
plt.show()

# Display pixel values for the first image
# image, label = train_dataset[0]
# print(f"Label: {label}")
# print(f"Image Shape: {image.shape}")
# print("Pixel values")
# print(image)

# Creating a simple tensorflow model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# Compile the model
model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print('Model are ready!')

#///////////////////////////Pytorch////////////////////////////////////////////////
# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2D(3, 32, kernel_size=3, activation='relu'),
        self.pool = nn.MaxPooling2D((2, 2)),
        self.fc1 = nn.Linear(32 * 15 * 15 * 128),
        self.fc2 = nn.Linear(128, 10),
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 15 * 15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
print("Pytorch model are ready")        

# Alternative with Sequential (if you want Keras-like style):
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x