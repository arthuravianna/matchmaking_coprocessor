import numpy as np

class MultiHeadSelfAttention2D:
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention2D, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads  # Dimension per head

        # Initialize weight matrices
        self.W_q = np.random.randn(embed_size, embed_size)
        self.W_k = np.random.randn(embed_size, embed_size)
        self.W_v = np.random.randn(embed_size, embed_size)
        #self.W_out = np.random.randn(embed_size, embed_size)

    def softmax(self, x):
        """ Numerically stable softmax function """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def relu(self, x):
        """ ReLU activation function """
        return np.maximum(0, x)

    def forward(self, x):
        batch_size, embed_size = x.shape  # Input is now (batch_size, embed_size)

        # Compute Q, K, V
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Split into multiple heads (batch_size, num_heads, head_dim)
        Q = Q.reshape(batch_size, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, self.num_heads, self.head_dim)

        # Scaled Dot-Product Attention
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
        attention_weights = self.softmax(scores)
        attention_output = np.matmul(attention_weights, V)

        # Concatenate attention heads back
        attention_output = attention_output.reshape(batch_size, embed_size)

        # Final projection + ReLU activation

        # ADC, Jungle, Middle, Support, Top
        W = np.array([0.2, 0.25, 0.22, 0.15, 0.18])

        #output = np.dot(attention_output, self.W_out)
        output = np.dot(W, attention_output)
        output = self.relu(output)  # Apply ReLU

        return output

    def update_w(self, Delta):
        for i in range(len(self.W_q)):
            for j in range(len(self.W_q[0])):
                self.W_q[i][j] = self.W_q[i][j] + Delta[i][j]
                self.W_k[i][j] = self.W_k[i][j] + Delta[i][j]
                self.W_v[i][j] = self.W_v[i][j] + Delta[i][j]


    def print_W_q(self):
        print("W_q:")
        print(self.W_q)

    def get_weights(self):
        return self.W_q, self.W_k, self.W_v


if __name__ == "__main__":
    # Example Usage
    batch_size, embed_size, num_heads = 5, 64, 4
    x = np.random.randn(batch_size, embed_size)  # Random input tensor (2D)

    multihead_attention = MultiHeadSelfAttention2D(embed_size, num_heads)
    output = multihead_attention.forward(x)

    print("Input shape:", x.shape)  # (batch_size, embed_size)
    print("Output shape:", output.shape)  # Should match input shape
    print(output)
