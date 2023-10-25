"""
This file hosues the implementation of the encoder component of the transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Final encoder model
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_encoder):
        """
        d_model: numebr of embeddings
        ffn_hidden: maximum number of neurons in the feed forward neural network
        num_heads: number of self-attention heads
        drop_prob: probability of dropping neurons
        num_encoder: number of encoder units
        """
        super().__init__
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_encoder)])
    def forward(self, x):
        # predict output
        x = self.layers(x)
        return x

# Encoder layer class
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # Input value that's untouched and will be added to normalization layer
        residual_x = x
        print("------- ATTENTION 1 ------")
        # Calcualte self-attention matrix
        x = self.attention(x, mask=None)
        print("------- DROPOUT 1 ------")
        # Randomly drop out neurons
        x = self.dropout1(x)
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        # Add residual_x to x and normalize
        x = self.norm1(x + residual_x)
        # Create another untouched residual input x
        residual_x = x
        print("------- ATTENTION 2 ------")
        # Pass to the feedforward neural network
        x = self.ffn(x)
        print("------- DROPOUT 2 ------")
        # Randomly drop out neurons again
        x = self.dropout2(x)
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        # Add residual_x to the x and normalize again
        x = self.norm2(x + residual_x)
        # Output of encoder layer
        return x

def single_head_attention(q, k, v, mask = None):
    # attention(q, k, v) = softmax(qK.T/sqrt(dk)V)
    # mask only for completeness
    # q, k, v = 30 x 8 x 200 x 64
    d_k = q.size()[-1] # 64
    # Only transpose the last 2 dimensions, because the first dimension is the batch size
    # scale the value with square root of d_k which is a constant value
    val_before_softmax = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(d_k)
    attention = F.softmax(val_before_softmax, dim = -1)
    # Multiply attention matrix with value matrix
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model              # for example 512
        self.num_heads = num_heads          # for example for 8 heads
        self.head_dim = d_model // num_heads    # head_dim will be 64
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)    # 512 x 1536
        self.linear_layer = nn.Linear(d_model, d_model)     # 512 x 512

    def forward(self, x, mask = None):
        batch_size, sequence_length, d_model = x.size()     # for example 30 x 200 x 512
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        q, k, v = qkv.chunk(3, dim = -1) # each are 30 x 8 x 200 x 64
        
    





if __name__ == "__init__":
    ...