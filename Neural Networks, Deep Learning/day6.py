import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Define Transformation
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Note: Added commas for tuple
])

# Load the dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms, download=True)

# Create the data loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Visualize dataset size 
print(f"Training data size: {len(train_dataset)}")
print(f"Test data size: {len(test_dataset)}")

# Define the model - CORRECTED VERSION
class NeuralNetwork(nn.Module): 
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input: 784, Output: 128
        self.fc2 = nn.Linear(128, 64)       # Input: 128, Output: 64
        self.fc3 = nn.Linear(64, 10)        # Input: 64, Output: 10
        
    def forward(self, X):
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))  # Changed from 'x' to 'X' and 'fc2'
        X = self.fc3(X)
        return X
    
model = NeuralNetwork()
print(model)  

# Define loss function and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

# Train the model - CORRECTED VERSION
def train_model(model, train_loader, criterion, optimizer, epochs=10): 
    model.train()
    for epoch in range(epochs):  # Changed variable name from 'epochs' to 'epoch'
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimizers
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()  # Fixed: changed loss_item() to loss.item()
        print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")  
        
train_model(model, train_loader, criterion, optimizer, epochs=10)  

# Evaluating the model
def Evaluating_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct/total:.4f}")
    
Evaluating_model(model, test_loader)

# Save the model
torch.save(model.state_dict(), '/home/Kellspell/Py-Dev/Neural Networks, Deep Learning/mnist_model.pth')

# Reload the model to test
loaded_model = NeuralNetwork()
loaded_model.load_state_dict(torch.load('/home/Kellspell/Py-Dev/Neural Networks, Deep Learning/mnist_model.pth'))
loaded_model.eval()  # Good practice to set to eval mode

# Verify loaded model performance
Evaluating_model(loaded_model, test_loader)