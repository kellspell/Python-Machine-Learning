import numpy as np

# Define query, key and value metrices
def generate_data(seq_len, embed_dim):
    np.random.seed(42)
    return np.random.rand(seq_len, embed_dim)

sequence_length = 4
embedding_dim = 3
query = generate_data(sequence_length, embedding_dim)
key = generate_data(sequence_length, embedding_dim)
value = generate_data(sequence_length, embedding_dim)

# Compute the Attention Score
scores = np.dot(query, key.T) / np.sqrt(embedding_dim)

# Apply softmax to normalize scores
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)

# Cpmpute the contaxt vector
context = np.dot(attention_weights, value) 

print("Attention weights: \n", attention_weights)
print("Context vector: \n ", context)
