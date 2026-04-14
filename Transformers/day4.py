import numpy as np
import matplotlib.pyplot as plt

# Define positional encoding 
def positional_encoding(seq_len, embed_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    angle_rate = 1 / np.power(10000, (2 * (i // 2 )) / np.float32(embed_dim))
    angle_rads = pos * angle_rate
    
    # Apply sine to even  indices and cosine to odd indices 
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return pos_encoding

# Generate position encoding 
seq_len = 100
embed_dim = 32
pos_encoding = positional_encoding(seq_len, embed_dim)

# Visualize positional encoding 
plt.figure(figsize=(10,6))
plt.pcolormesh(pos_encoding, cmap='viridis')
plt.colorbar()
plt.title('Positional Encoding')
plt.xlabel('Embedding Dimention')
plt.ylabel('Position')
plt.show()    
 
 


