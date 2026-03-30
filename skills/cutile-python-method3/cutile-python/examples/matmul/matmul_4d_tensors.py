"""Example 4: matrix multiplication with 4D tensors - note the difference from 3D tensors"""
import torch
import math
import cuda.tile as ct


# A has shape (P, Q, M, K)
# B has shape (P, Q, K, N)
# output has shape (P, Q, M, N)
@ct.kernel
def matmul_kernel(
    A,
    B,
    output,
    BLOCK_M: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    bid_p = ct.bid(0) // A.shape[1]
    bid_q = ct.bid(0) % A.shape[1]
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    for k in range(A.shape[3] // BLOCK_K):
        # Load tiles from A and B
        a = ct.load(A, index=(bid_p, bid_q, bid_m, k), shape=(1, 1, BLOCK_M, BLOCK_K))
        b = ct.load(B, index=(bid_p, bid_q, k, bid_n), shape=(1, 1, BLOCK_K, BLOCK_N))
        a = ct.reshape(a, (BLOCK_M, BLOCK_K))
        b = ct.reshape(b, (BLOCK_K, BLOCK_N))
        # mma supports 2D or 3D tensors, so we need to reshape acc
        acc = ct.mma(a, b, acc)

    # Optional: convert acc to output dtype when the output dtype is not float32
    acc = ct.astype(acc, output.dtype)
    # Reshape acc to (1, 1, BLOCK_M, BLOCK_N) to match the shape of output
    acc = ct.reshape(acc, (1, 1, BLOCK_M, BLOCK_N))
    # Store result in output
    ct.store(output, index=(bid_p, bid_q, bid_m, bid_n), tile=acc)


def reference_matmul(A, B):
    return torch.matmul(A, B)


if __name__ == "__main__":
    P = 11
    Q = 5
    M = 1024
    K = 1024
    N = 512
    A = torch.rand(P, Q, M, K, dtype=torch.float16, device="cuda")
    B = torch.rand(P, Q, K, N, dtype=torch.float16, device="cuda")
    cutile_output = torch.zeros(P, Q, M, N, dtype=torch.float16, device="cuda")

    BLOCK_M = 16
    BLOCK_K = 64
    BLOCK_N = 32
    ## grid supports max 3 dimensions
    grid = (P * Q, math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N))
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, cutile_output, BLOCK_M, BLOCK_K, BLOCK_N))
    reference_output = reference_matmul(A, B)

    assert torch.allclose(cutile_output, reference_output, atol=1e-2, rtol=1e-2)
    print("Test passed!")
