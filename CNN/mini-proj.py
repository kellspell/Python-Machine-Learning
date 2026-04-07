import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# Define Transformation - FIXED for grayscale MNIST
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
    
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
    
])

# Load the Cifar10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

# Create the data loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) 

# Visualize dataset size 
print(f"Training data size: {len(train_dataset)}")
print(f"Test data size: {len(test_dataset)}")

# Define the CNN model - FIXED for 1-channel input
class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.5)
        
        # After two conv layers with pooling, 28x28 becomes:
        # 28 -> conv1 (5x5) -> 24 -> pool -> 12
        # 12 -> conv2 (5x5) -> 8 -> pool -> 4
        # So final size is 16 * 4 * 4 = 256
        
        self._calculation_conv_output()
        
        self.fc1 = nn.Linear(self.conv_output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def _calculation_conv_output(self):
        # Dummy input tensor with the same size as input images 
        dummy_input = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            output = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(dummy_input))))))) 
        self.conv_output_size = output.numel()      
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
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
        epoch_loss = running_loss / len(train_loader)
        training_loss.append(epoch_loss)    
        
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

train_model(model, train_loader, criterion, optimizer)        
        
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