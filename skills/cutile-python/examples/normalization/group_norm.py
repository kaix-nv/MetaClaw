"""Test 1: This test below implements Group Normalization with bias."""
import torch
import cuda.tile as ct


@ct.kernel
def group_norm_kernel(
    input,
    output,
    weight,
    bias,
    C_per_group: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    eps: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
):
    """
    cuTile Group Normalization kernel implementation.
        This kernel is designed to handle the case where C, H, W are large.

    Args:
        input: Input tensor of shape (N, C, H, W)
        output: Output tensor of shape (N, C, H, W)
        weight: Weight tensor of shape (C,)
        bias: Bias tensor of shape (C,)
        N: Number of input tensors
        C: Number of channels
        H: Height
        W: Width
        C_per_group: Number of channels per group
        eps: Epsilon value for numerical stability
    """
    bid_n = ct.bid(0)
    bid_g = ct.bid(1)
    size = H * W * C_per_group

    # Step 1: Compute mean
    tx_sum = ct.full((1, 1, 1), 0.0, dtype=torch.float32)
    for i in range(size // BLOCK_SIZE):
        tx = ct.load(input, index=(bid_n, bid_g, i), shape=(1, 1, BLOCK_SIZE))
        tx_sum = tx_sum + ct.sum(tx, axis=2, keepdims=True)
    tx_mean = tx_sum / size

    # Step 2: Compute variance
    tx_var = ct.full((1, 1, 1), 0.0, dtype=torch.float32)
    for i in range(size // BLOCK_SIZE):
        tx = ct.load(input, index=(bid_n, bid_g, i), shape=(1, 1, BLOCK_SIZE))
        tx_var = tx_var + ct.sum(ct.pow(tx - tx_mean, 2), axis=2, keepdims=True)
    tx_var = tx_var / size

    # Step 3: Normalize and apply affine transformation
    tw = ct.load(weight, index=(bid_g,), shape=(1,))
    tb = ct.load(bias, index=(bid_g,), shape=(1,))
    for i in range(size // BLOCK_SIZE):
        tx = ct.load(input, index=(bid_n, bid_g, i), shape=(1, 1, BLOCK_SIZE))
        tx_norm = (tx - tx_mean) / ct.sqrt(tx_var + eps)
        result = tx_norm * tw + tb
        result = result.astype(output.dtype)
        ct.store(output, index=(bid_n, bid_g, i), tile=result)


def cutile_groupnorm(input, weight, bias, eps=1e-5):
    N, C, H, W = input.shape
    C_per_group = C // num_groups
    input = input.view(N, num_groups, -1)
    grid = (N, num_groups, 1)
    BLOCK_SIZE = 128
    output = torch.zeros_like(input)
    ct.launch(torch.cuda.current_stream(), grid, group_norm_kernel, (
        input, output, weight, bias, C_per_group, H, W, eps, BLOCK_SIZE
    ))
    return output.view(N, C, H, W)


def pytorch_reference(model, input):
    return model(input)


if __name__ == "__main__":
    torch.manual_seed(42)
    N, C, H, W = 4, 64, 32, 64
    num_groups = 8
    dtype = torch.float16

    assert (
        C % num_groups == 0
    ), f"Number of channels ({C}) must be divisible by num_groups ({num_groups})"
    assert dtype == torch.float16, "Only float16 is supported"

    class GroupNorm(torch.nn.Module):
        def __init__(self, num_groups, num_channels):
            super(GroupNorm, self).__init__()
            self.group_norm = torch.nn.GroupNorm(num_groups, num_channels)

        def forward(self, x):
            return self.group_norm(x)

    model = GroupNorm(num_groups, C).eval().to(dtype).to("cuda")

    # Create test input
    input_tensor = torch.rand(N, C, H, W, dtype=dtype, device="cuda")
    weight = model.group_norm.weight.data
    bias = model.group_norm.bias.data

    eps = 1e-5
    output_cutile = cutile_groupnorm(input_tensor, weight, bias, eps)

    # Test against PyTorch's built-in GroupNorm
    output_pytorch = pytorch_reference(model, input_tensor)

    # Validate results
    if torch.allclose(output_cutile, output_pytorch, atol=1e-2, rtol=1e-2):
        print("Test passed!")
    else:
        print("Test failed!")
        abs_diff = torch.abs(output_cutile - output_pytorch)
        print(f"Max absolute difference: {torch.max(abs_diff).item():.6f}")
        print(f"Mean absolute difference: {torch.mean(abs_diff).item():.6f}")
