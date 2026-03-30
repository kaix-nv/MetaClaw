"""Case 2: Flash Attention without Causal Mask with the input shape (B, H, S, D)"""
import torch
import numpy as np
import cuda.tile as ct

@ct.kernel
def fmha_no_causal_mask_kernel(
    Q, K, V, O,
    sm_scale,
    batch_size: ct.Constant[int], num_heads: ct.Constant[int], seq_len: ct.Constant[int], head_dim: ct.Constant[int],
    BLOCK_M: ct.Constant[int], BLOCK_N: ct.Constant[int], BLOCK_D: ct.Constant[int]
):
    # Get program IDs
    bid_batch = ct.bid(0)
    bid_head = ct.bid(1)
    bid_seq = ct.bid(2)

    # Load Q block
    q = ct.load(Q, index=(bid_batch, bid_head, bid_seq, 0), shape=(1, 1, BLOCK_M, BLOCK_D))
    q = ct.reshape(q, (BLOCK_M, BLOCK_D))

    # Initialize output accumulators
    o = ct.full((BLOCK_M, BLOCK_D), 0.0, dtype=np.float32)
    m = ct.full((BLOCK_M, 1), -np.inf, dtype=np.float32)  # Max for softmax
    l = ct.full((BLOCK_M, 1), 0.0, dtype=np.float32)  # Sum for softmax normalization

    # Iterate over K, V blocks (causal attention)
    for start_n in range((seq_len + BLOCK_N - 1) // BLOCK_N):
        # Load K block, shape: (1, 1, BLOCK_N, BLOCK_D)
        k = ct.load(K, index=(bid_batch, bid_head, start_n, 0), shape=(1, 1, BLOCK_N, BLOCK_D))
        k = ct.reshape(k, (BLOCK_N, BLOCK_D))
        k_t = ct.transpose(k, 1, 0)

        # Compute Q(K^T), shape: (BLOCK_M, BLOCK_N)
        qk = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
        qk = ct.mma(q, k_t, qk)

        # Apply scale
        qk = qk * sm_scale

        # Compute softmax
        m_i = ct.max(qk, axis=1, keepdims=True)

        # Update running max and sum
        m_new = ct.maximum(m, m_i)
        p = ct.exp(qk - m_new)
        l_i = ct.sum(p, axis=1, keepdims=True)

        alpha = ct.exp(m - m_new)
        l = l * alpha + l_i
        o = o * alpha

        # Load V block, shape: (1, 1, BLOCK_N, BLOCK_D)
        v = ct.load(V, index=(bid_batch, bid_head, start_n, 0), shape=(1, 1, BLOCK_N, BLOCK_D))
        v = ct.reshape(v, (BLOCK_N, BLOCK_D))
        v = ct.astype(v, np.float32)  # match dtype of p (float32) for ct.mma

        # Update output
        o = ct.mma(p, v, o)

        m = m_new

    # Normalize output
    o = o / l
    o = ct.reshape(o, (1, 1, BLOCK_M, BLOCK_D))
    o = ct.astype(o, O.dtype)

    # Write output
    ct.store(O, index=(bid_batch, bid_head, bid_seq, 0), tile=o)

def fmha(q, k, v, sm_scale):
    batch_size, num_heads, seq_len, head_dimension = q.shape
    assert k.shape == v.shape == (batch_size, num_heads, seq_len, head_dimension)

    # Allocate output
    o = torch.empty_like(q)

    # Grid dimensions
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = head_dimension
    # BLOCK_SIZE: (1, BLOCK_M, 1, BLOCK_D)
    grid = (batch_size, num_heads, (seq_len + BLOCK_M - 1) // BLOCK_M)
    print(f"Grid: {grid}")

    # Launch kernel
    ct.launch(torch.cuda.current_stream(), grid, fmha_no_causal_mask_kernel, (
        q, k, v, o,
        sm_scale,
        batch_size, num_heads, seq_len, head_dimension,
        BLOCK_M, BLOCK_N, BLOCK_D)
    )
    return o

# Example usage
batch_size, seq_len, num_heads, head_dimension = 2, 1024, 8, 64
q = torch.rand(batch_size, num_heads, seq_len, head_dimension, device='cuda', dtype=torch.float16)
k = torch.rand(batch_size, num_heads, seq_len, head_dimension, device='cuda', dtype=torch.float16)
v = torch.rand(batch_size, num_heads, seq_len, head_dimension, device='cuda', dtype=torch.float16)
sm_scale = 1.0 / (head_dimension ** 0.5)

# Run cuTile implementation
output = fmha(q, k, v, sm_scale)

# Run PyTorch implementation for validation
def pytorch_fmha(q, k, v, sm_scale):
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Apply softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # Compute weighted sum of values
    output = torch.matmul(attn_weights, v)

    # output = output.transpose(1, 2)
    return output

# Run PyTorch implementation
pytorch_output = pytorch_fmha(q, k, v, sm_scale)

# Compare results
is_close = torch.allclose(output, pytorch_output, rtol=1e-2, atol=1e-2)
if is_close:
    print(f"✅ Test passed!")
    print(f"Outputs are close within tolerance")
else:
    print(f"❌ Validation FAILED")
    print(f"Outputs differ beyond tolerance")
    print(f"Max absolute difference: {torch.max(torch.abs(output - pytorch_output)):.6f}")
    print(f"Mean absolute difference: {torch.mean(torch.abs(output - pytorch_output)):.6f}")
