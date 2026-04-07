import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Define Transformation - FIXED for grayscale MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Single value for grayscale
])

# Load the dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create the data loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Visualize dataset size 
print(f"Training data size: {len(train_dataset)}")
print(f"Test data size: {len(test_dataset)}")

# Define the CNN model - FIXED for 1-channel input
class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Changed from 3 to 1 channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # After two conv layers with pooling, 28x28 becomes:
        # 28 -> conv1 (5x5) -> 24 -> pool -> 12
        # 12 -> conv2 (5x5) -> 8 -> pool -> 4
        # So final size is 16 * 4 * 4 = 256
        self.fc1 = nn.Linear(16 * 4 * 4, 128)  # Fixed size calculation
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):  # Fixed variable name consistency
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # Fixed size
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  
        x = self.fc3(x)
        return x
    
model = CNN()
print(model) 

# Define Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 

# Train the model - FIXED
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Starting from Zero gradient
            optimizer.zero_grad()
             
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backwards pass and optimizer - FIXED case
            loss.backward()  # Was 'Backwards()'
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation loop
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.4f}%")  

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=10)

# Evaluate the model
evaluate_model(model, test_loader)