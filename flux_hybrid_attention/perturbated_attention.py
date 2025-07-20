import torch
import torch.nn.functional as F
import math


def perturbated_attention(q, k, v, alpha, beta):
    """
    Computes a custom attention mechanism that blends standard scaled-dot product attention
    with an identity matrix.

    (beta * softmax(Q * K^T / sqrt(d_k)) + (1 - beta) * I / alpha) * V

    Args:
        q (torch.Tensor): Query tensor. Shape: (batch_size, num_heads, seq_len_q, head_dim)
        k (torch.Tensor): Key tensor. Shape: (batch_size, num_heads, seq_len_k, head_dim)
        v (torch.Tensor): Value tensor. Shape: (batch_size, num_heads, seq_len_v, head_dim)
                          Note: seq_len_k and seq_len_v must be the same.
        alpha (float): A scaling factor for the identity matrix.
        beta (float): A hyperparameter between 0 and 1 to control the blend.

    Returns:
        torch.Tensor: The output of the attention mechanism.
                      Shape: (batch_size, num_heads, seq_len_q, head_dim)
    """

    # Get the dimension of the key vectors
    d_k = k.size(-1)

    # 1. Calculate the standard scaled dot-product attention scores
    # (Q * K^T) / sqrt(d_k)
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. Apply softmax to get the standard attention weights
    standard_attention_weights = F.softmax(attention_scores, dim=-1)

    # 3. Create the identity matrix
    # It needs to have the same shape as the attention_scores matrix for broadcasting
    seq_len_q = q.size(-2)
    seq_len_k = k.size(-2)

    if seq_len_q != seq_len_k:
        raise ValueError(
            "Sequence length of query and key must be equal to use an identity matrix."
        )

    # Create an identity matrix of size (seq_len_q, seq_len_k)
    identity_matrix = torch.eye(seq_len_q, device=q.device, dtype=q.dtype)

    # 4. Blend the standard attention weights with the identity matrix using beta
    attention_weights = (
        beta * standard_attention_weights + (1 - beta) * identity_matrix / alpha
    )

    # 5. Multiply the weights by the Value tensor (V)
    output = torch.matmul(attention_weights, v)

    return output
