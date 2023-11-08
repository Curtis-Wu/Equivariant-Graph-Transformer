"""
This file hosues the implementation of the encoder component of the transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Final encoder model
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, dropout_r, num_ffn, num_encoder):
        """
        d_model: numebr of embeddings
        num_heads: number of self-attention heads
        dropout_r: dropout probability/rate
        num_ffn: number of neurons in the feed forward neural network
        num_encoder: number of encoder units
        """
        super().__init__
        # Create num_encoder encoder layers
        self.layers = nn.Sequential(*[EncoderLayer(d_model, num_heads, dropout_r, num_ffn)
                                      for _ in range(num_encoder)])
    def forward(self, x):
        x = self.layers(x)
        return x


# Encoder layer class
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_ffn, dropout_r):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(p = dropout_r)
        self.ffn = MLP(d_model = d_model, hidden = num_ffn, dropout_r = dropout_r)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout(p = dropout_r)

    def forward(self, x):
        # Input value that's untouched and will be added to normalization layer
        residual_x = x
        # Calcualte self-attention matrix
        x = self.attention(x, mask=None)
        # Dropout
        x = self.dropout1(x)
        # Add residual_x to x and normalize
        x = self.norm1(x + residual_x)
        # Create another untouched residual input x
        residual_x = x
        # Pass to the feedforward neural network
        x = self.ffn(x)
        # Dropout
        x = self.dropout2(x)
        # Add residual_x to the x and normalize again
        x = self.norm2(x + residual_x)
        # Output of encoder layer
        return x


def single_head_attention(q, k, v):
    # attention(q, k, v) = softmax(qK.T/sqrt(dk)V)
    d_k = q.size()[-1] # 64
    # Only transpose the last 2 dimensions, because the first dimension is the batch size
    # scale the value with square root of d_k which is a constant value
    val_before_softmax = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(d_k)
    attention = F.softmax(val_before_softmax, dim = -1) # 200 x 200
    # Multiply attention matrix with value matrix
    values = torch.matmul(attention, v) # 200 x 64
    return values, attention


# Multi-head attention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model                              # for example 512
        self.num_heads = num_heads                          # for example for 8 heads
        self.head_dim = d_model // num_heads                # head_dim will be 64
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)    # 512 x 1536
        self.linear_layer = nn.Linear(d_model, d_model)     # 512 x 512

    def forward(self, x):
        batch_size, sequence_length, d_model = x.size()     # for example 30 x 200 x 512
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        q, k, v = qkv.chunk(3, dim = -1) # breakup using the last dimension, each are 30 x 8 x 200 x 64

        values, attention = single_head_attention(q, k, v)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)

        return out
        
# Layer normalization class
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps = 1e-8):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps # to take care of zero division
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # learnable parameter "std" (512,)
        self.beta = nn.Parameter(torch.zeros(parameters_shape)) # learnable parameter "mean" (512,)

    def forward(self, inputs):
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim = dims, keepdim = True) # eg. for (30, 200, 512) inputs, mean -> (30, 200, 1)
        var = ((inputs - mean)**2).mean(dim = dims, keepdim = True) # (30, 200, 1) 
        std = (var + self.eps).sqrt() # (30, 200, 1)
        y = (inputs - mean) / std # Normalized output (30, 200, 512)
        out = self.gamma*y + self.beta # Apply learnable parameters

        return out


class MLP(nn.Module):
    def __init__(self, d_model, hidden, dropout_r = 0.1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout_r)

    def forward(self, x):
        x = self.linear1(self.relu(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x