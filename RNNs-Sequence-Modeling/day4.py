import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU

vocab_size = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size) # what we are saying here is that from the dataset we only want to use is the first 10000 words

# Let's preprocessing the data
X_train = pad_sequences(X_train, maxlen=max_len, padding='post' )
X_test = pad_sequences(X_test, maxlen=max_len, padding='post' )

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Creating a model
gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    GRU(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

gru_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build the model with the input shape
gru_model.build(input_shape=(None, max_len))  # None = batch size, max_len = sequence length

gru_model.summary()

history = gru_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
loss, accuracy = gru_model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")