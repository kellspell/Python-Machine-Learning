import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Check assert
        assert embed_dim % num_heads == 0, "Embedding dimention must be divisibel by number of heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size = x.size (0)
        # Linear projections 
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)       
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute  attention score
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute context
        context = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out(context), attention_weights
    
# Sample Input
seq_len, embed_dim = 4, 8
x = torch.rand(1, seq_len, embed_dim)

# instantiate and test
mha = MultiHeadAttention(embed_dim, num_heads=2)
context, attn_weights = mha(x) 

print("Attention weights: \n", attn_weights)
print("Context vector: \n ", context) 
        