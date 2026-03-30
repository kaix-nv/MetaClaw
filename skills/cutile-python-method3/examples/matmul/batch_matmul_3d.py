"""Example 3: batch matrix multiplication (BMM) with 3D tensors"""
import torch
import math
import cuda.tile as ct


# A has shape (B, M, K)
# B has shape (B, K, N)
# output has shape (B, M, N)
@ct.kernel
def matmul_kernel(
    A,
    B,
    output,
    BLOCK_M: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    bid_b = ct.bid(0)
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)

    acc = ct.full((1, BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    for k in range(A.shape[2] // BLOCK_K):
        # Load tiles from A and B
        a = ct.load(A, index=(bid_b, bid_m, k), shape=(1, BLOCK_M, BLOCK_K))
        b = ct.load(B, index=(bid_b, k, bid_n), shape=(1, BLOCK_K, BLOCK_N))
        # Compute matrix multiplication
        acc = ct.mma(a, b, acc)

    # Optional: convert acc to output dtype when the output dtype is not float32
    acc = ct.astype(acc, output.dtype)
    # Store result in output
    ct.store(output, index=(bid_b, bid_m, bid_n), tile=acc)


def reference_matmul(A, B):
    return torch.bmm(A, B)


if __name__ == "__main__":
    BATCH_SIZE = 11
    M = 1024
    K = 1024
    N = 512
    A = torch.rand(BATCH_SIZE, M, K, dtype=torch.float16, device="cuda")
    B = torch.rand(BATCH_SIZE, K, N, dtype=torch.float16, device="cuda")
    cutile_output = torch.zeros(BATCH_SIZE, M, N, dtype=torch.float16, device="cuda")

    BLOCK_M = 16
    BLOCK_K = 64
    BLOCK_N = 32
    grid = (BATCH_SIZE, math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N))
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, cutile_output, BLOCK_M, BLOCK_K, BLOCK_N))
    reference_output = reference_matmul(A, B)

    assert torch.allclose(cutile_output, reference_output, atol=1e-2, rtol=1e-2)
    print("Test passed!")
