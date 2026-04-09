import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

vocab_size = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size) # what we are saying here is that from the dataset we only want to use is the first 10000 words

# decode review to text for preprocessing 
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
decoded_reviews = [''.join([reverse_word_index.get(i - 3, '?' ) for i in review]) for review in X_train[:5]]

# Let's preprocessing the data
X_train = pad_sequences(X_train, maxlen=max_len, padding='post' )
X_test = pad_sequences(X_test, maxlen=max_len, padding='post' )

# print(f"Training data shape: {X_train.shape}")
# print(f"Test data shape: {X_test.shape}")

# Load a pre-trained GloVe embeddings
# !The code below word but there is an issue with the dataset , so we are going to use a different one!
# embeddings_index = {}
# glove_db = '/home/Kellspell/Py-Dev/RNNs-Sequence-Modeling/embeddings.txt'
# with open(glove_db, 'r',encoding='utf-8') as file:
#     for line in file:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
# print(f"Loaded: {len(embeddings_index)}, words vectors") 

embeddings_index = {}
glove_db = '/home/Kellspell/Py-Dev/RNNs-Sequence-Modeling/embeddings.txt'

stats = {'total': 0, 'loaded': 0, 'errors': 0}

with open(glove_db, 'r', encoding='utf-8') as file:
    for line in file:
        stats['total'] += 1
        values = line.split()
        
        if len(values) < 2:
            continue
            
        word = values[0]
        
        # Skip period word
        if word == '.':
            continue
            
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            stats['loaded'] += 1
        except ValueError:
            stats['errors'] += 1

print(f"Total lines:     {stats['total']:,}")
print(f"Loaded vectors:  {stats['loaded']:,}")
print(f"Skipped errors:  {stats['errors']:,}")
print(f"Success rate:    {stats['loaded']/stats['total']*100:.2f}%")    

# Preparing embedding matrix
embeddings_dim = 100
embeddings_mx = np.zeros((vocab_size, embeddings_dim))

for word, i in word_index.items():
    if i < vocab_size:
        embeddings_vector = embeddings_index.get(word)
        if embeddings_vector is not None:
            # Take only the first 100 dimensions
            embeddings_mx[i] = embeddings_vector[:embeddings_dim]
            
# Difine the LSTM model wth GloVe embeddings
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embeddings_dim,
        weights=[embeddings_mx],
        trainable=False
    ),
    LSTM(
        128,
        activation='tanh',
        return_sequences=False
    ),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer= 'adam',
    loss='binary_crossentropy',
    metrics=['accuracy'] 
)  

model.summary()  

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64,
    verbose=1
)

# Evaluate
loss, accuracy =  model.evaluate(X_test, y_test, verbose=1)
print(f"LSTM model with GloVe test accuracy{accuracy:.4f}")      
            
