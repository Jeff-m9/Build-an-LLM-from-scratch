import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]
    ]
)

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
# print(attn_scores_2)

attn_weights_2_temp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_temp)
# print("Sum:", attn_weights_2_temp.sum())

# Simple application of softmaxing
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())

# Softmaxing using pytorch-softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

# Calculating the context vector
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
# print(context_vec_2)


# Finding the attention scores for all inputs
attn_scores = inputs @ inputs.T
# print(attn_scores)

# Normalizing the attention scores to get the weights
attn_weights = torch.softmax(attn_scores, dim=-1)
# print(attn_weights)

all_context_vec = attn_weights @ inputs
print(all_context_vec)
