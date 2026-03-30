"""Split-k GEMM implementation in cuTile"""
import torch
import cuda.tile as ct

@ct.kernel
def split_gemm(
    A,
    B,
    C,
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    SPLIT_K: ct.Constant[int],
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    bid_k = ct.bid(2)

    start_k = SPLIT_K * bid_k
    end_k = start_k + SPLIT_K
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    for k in range(start_k, end_k):
        a_tile = ct.load(A, index=(bid_m, k), shape=(BLOCK_M, BLOCK_K), allow_tma=False)
        b_tile = ct.load(B, index=(k, bid_n), shape=(BLOCK_K, BLOCK_N), allow_tma=False)
        acc = ct.mma(a_tile, b_tile, acc)

    # Per-dimension index tuple required for rank-2 arrays in v1.2.0
    offset_m = bid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
    offset_n = bid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)

    ct.atomic_add(C, (offset_m[:, None], offset_n[None, :]), acc)


def launch_split_gemm(A, B, C):
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 64
    SPLIT_K = 4

    grid_x = (A.shape[0] + BLOCK_M - 1) // BLOCK_M
    grid_y = (B.shape[1] + BLOCK_N - 1) // BLOCK_N
    grid_z = (A.shape[1] + BLOCK_K - 1) // (BLOCK_K * SPLIT_K)

    grid = (grid_x, grid_y, grid_z)

    ## split_gemm[grid](A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        split_gemm,
        (A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K),
    )
    return C


def reference_gemm(A, B):
    return torch.matmul(A, B)


def main():
    A = torch.rand(512, 10240, dtype=torch.float32, device="cuda")
    B = torch.rand(10240, 256, dtype=torch.float32, device="cuda")

    # Test cuda.tile implementations
    C_split = torch.zeros(512, 256, dtype=torch.float32, device="cuda")
    launch_split_gemm(A, B, C_split)

    C_ref = reference_gemm(A, B)

    # Verification
    print("=== Correctness Verification ===")

    verified = torch.allclose(C_split, C_ref, atol=1e-1, rtol=1e-1)
    if verified:
        print("Test passed! cuda.tile Split GEMM verified")
    else:
        print("cuda.tile Split GEMM failed")
        print(f"Max error: {torch.max(torch.abs(C_split - C_ref))}")


if __name__ == "__main__":
    main()
