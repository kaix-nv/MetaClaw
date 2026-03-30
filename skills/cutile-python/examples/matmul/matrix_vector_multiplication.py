"""Example 2: matrix vector multiplication"""
import torch
import math
import cuda.tile as ct


# A has shape (M, K)
# B has shape (K)
# output has shape (M)
@ct.kernel
def cutile_gemv_kernel(
    A,
    B,
    output,
    BLOCK_M: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    bid_m = ct.bid(0)
    acc = ct.full((BLOCK_M, 1), 0.0, dtype=ct.float32)
    for k in range(A.shape[1] // BLOCK_K):
        # Load tiles from A and B
        a = ct.load(A, index=(bid_m, k), shape=(BLOCK_M, BLOCK_K))
        b = ct.load(B, index=(k,), shape=(BLOCK_K,))
        # Need to reshape b to (BLOCK_K, 1) to match the shape of a
        b2 = ct.reshape(b, (BLOCK_K, 1))
        # Compute matrix multiplication
        acc = ct.mma(a, b2, acc)

    # Optional: convert acc to output dtype when the output dtype is not float32
    acc = ct.astype(acc, output.dtype)
    # Reshape acc to (BLOCK_M,) to match the shape of output
    acc = ct.reshape(acc, (BLOCK_M,))
    # Store result in C
    ct.store(output, index=(bid_m,), tile=acc)


def reference_matmul(A, B):
    return torch.matmul(A, B)


if __name__ == "__main__":
    M = 1024
    K = 1024
    A = torch.rand(M, K, dtype=torch.float16, device="cuda")
    B = torch.rand(K, dtype=torch.float16, device="cuda")
    cutile_output = torch.zeros(M, dtype=torch.float16, device="cuda")

    BLOCK_M = 16
    BLOCK_K = 64
    grid = (math.ceil(M / BLOCK_M), 1)
    ct.launch(torch.cuda.current_stream(), grid, cutile_gemv_kernel, (A, B, cutile_output, BLOCK_M, BLOCK_K, 1))
    reference_output = reference_matmul(A, B)

    assert torch.allclose(cutile_output, reference_output, atol=1e-2, rtol=1e-2)
    print("Test passed!")
