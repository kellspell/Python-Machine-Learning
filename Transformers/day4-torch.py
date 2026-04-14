import torch
import torch.nn as nn

class TransformersWithPositionEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len,  num_heads, ff_dim):
        super(TransformersWithPositionEncoding, self).__init__()
        self.embedding = nn.Embedding(seq_len, embed_dim)
        self.positional_encoding = nn.parameter(torch.tensor(positional_encoding(seq_len, embed_dim), dtype=torch.float32))
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Add Positional encoding to embedding
        x = self.embedding(x) + self.positional_encoding
        # Self Attention
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.norm1(x + attn_output)
        # Feed Forward Network
        ffn_output = self.fnn(x)
        x = self.norm2(x + ffn_output)
        return x  
    
# Define the model parameters
embed_dim = 16
seq_len = 50
num_heads = 4
ff_dim = 64

# Create the model
model = TransformersWithPositionEncoding(embed_dim, seq_len, num_heads, ff_dim) 
print(model)    
        