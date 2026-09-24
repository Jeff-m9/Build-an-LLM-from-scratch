import torch
from torch import nn

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)
d_in = inputs.shape[1]
d_out = 2

batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape)


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # Query, key, value weight matrices
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # b-> batch, num_tokens-> number of tokens per batch, d_in-> number of dimensions (2,6,2)
        b, num_tokens, d_in = x.shape

        # get the keys, queries and values by multiplying their respective weight matrices with the inputs
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # calculate the attention score, i.e, query * key transpose
        attn_scores = queries @ keys.transpose(1, 2)

        # mask the attention scores
        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf,  #':num_token' is used as an edge case to get the number of tokens currently being used
        )

        # normalize the attention scores to get the attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # apply dropout to the attention weights
        attn_weights = self.dropout(attn_weights)

        # calculate the context vector using values
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
# print("context_vecs.shape", context_vecs)
