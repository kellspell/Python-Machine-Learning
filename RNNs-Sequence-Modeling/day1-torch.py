import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import imdb # dataset for sentiment analysis
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size) # what we are saying here is that from the dataset we only want to use is the first 10000 words

# Let's preprocessing the data
X_train = pad_sequences(X_train, maxlen=max_len, padding='post' )
X_test = pad_sequences(X_test, maxlen=max_len, padding='post' )

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# convert numpy arrayes into pytorch tensors and creates the dataset of pair inputs and lables
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return torch.sigmoid(self.fc(hidden[-1]))

model = RNNModel(vocab_size=10000, embedding_dim=128, hidden_dim=128, output_dim=1)   

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 

# Train the model
def train_rnn(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0  # Changed from 'epochs_loss' to 'running_loss'
        for X_batch, y_batch in train_loader:
            # Starting from Zero gradient
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze(1)  # Changed from .sequence(1)
            loss = criterion(predictions, y_batch.float())  # Fixed: was criterion(criterion, ...)
            
            # Backwards pass and optimizer
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")   

# Train the model
train_rnn(model, train_loader, criterion, optimizer, epochs=5)  

# Evaluation loop
def evaluate_rnn(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test)).squeeze(1)
        loss = criterion(predictions, torch.tensor(y_test).float())
        accuracy = ((predictions > 0.5) == torch.tensor(y_test).float()).float().mean().item()
    print(f"Test Loss: {loss.item():.4f}, Test Accuracy{accuracy:.4f}") 
    
# Evaluate the model
evaluate_rnn(model, X_test, y_test)                   






















