import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention2D, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size  
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads  # Dimension per head

        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)

        # Final linear layer after concatenation
        self.fc_out = nn.Linear(embed_size, embed_size)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, embed_size = x.shape  # Now x is (batch_size, embed_size)

        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads (batch_size, num_heads, head_dim)
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate attention heads back
        attention_output = attention_output.view(batch_size, embed_size)

        # Final projection + ReLU activation
        output = self.fc_out(attention_output)
        
        # ADC, Jungle, Middle, Support, Top
        W = torch.Tensor([0.2, 0.25, 0.22, 0.15, 0.18])
        output = self.relu(W@output)  # Apply ReLU

        return output



if __name__ == "__main__":
    # Example input
    batch_size = 5 #2
    seq_length = 64 #5
    embed_size = 64
    num_heads = 4

    x = torch.rand(batch_size, seq_length)  # Random input tensor

    # Initialize and apply attention layer
    multihead_attention = MultiHeadSelfAttention2D(embed_size, num_heads)
    output = multihead_attention(x)

    print("Input shape:", x.shape)  # (batch_size, seq_length)
    print("Output shape:", output.shape)  # Should match input shape
    print(output)
