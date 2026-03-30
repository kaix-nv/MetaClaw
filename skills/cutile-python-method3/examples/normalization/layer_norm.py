"""Test 2: This test below implements Layer Normalization with bias."""
import torch
import cuda.tile as ct


@ct.kernel
def layer_norm_kernel(
    input,
    output,
    weight,
    bias,
    C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    eps: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
):
    """
    cuTile Layer Normalization kernel implementation.
        This kernel is designed to handle the case where C, H, W are large.

    Args:
        input: Input tensor of shape (N, C*H*W)
        output: Output tensor of shape (N, C*H*W)
        weight: Weight tensor of shape (1, C*H*W)
        bias: Bias tensor of shape (1, C*H*W)
        H: Height
        W: Width
        eps: Epsilon value for numerical stability
    """
    bid_n = ct.bid(0)
    size = C * H * W
    # Step 1: Compute mean
    tx_sum = ct.full((1, 1), 0.0, dtype=torch.float32)
    for i in range(size // BLOCK_SIZE):
        tx = ct.load(input, index=(bid_n, i), shape=(1, BLOCK_SIZE))
        tx_sum = tx_sum + ct.sum(tx, axis=1, keepdims=True)
    tx_mean = tx_sum / size

    # Step 2: Compute variance
    tx_var = ct.full((1, 1), 0.0, dtype=torch.float32)
    for i in range(size // BLOCK_SIZE):
        tx = ct.load(input, index=(bid_n, i), shape=(1, BLOCK_SIZE))
        tx_var = tx_var + ct.sum(ct.pow(tx - tx_mean, 2), axis=1, keepdims=True)
    tx_var = tx_var / size

    for i in range(size // BLOCK_SIZE):
        tx = ct.load(input, index=(bid_n, i), shape=(1, BLOCK_SIZE))
        tw = ct.load(weight, index=(0, i), shape=(1, BLOCK_SIZE))
        tb = ct.load(bias, index=(0, i), shape=(1, BLOCK_SIZE))
        normalized = (tx - tx_mean) / ct.sqrt(tx_var + eps)
        result = normalized * tw + tb
        result = result.astype(output.dtype)
        ct.store(output, index=(bid_n, i), tile=result)


def cutile_layernorm(input, weight, bias, eps=1e-5):
    N, C, H, W = input.shape
    input = input.view(N, -1)
    weight = weight.view(1, -1)
    bias = bias.view(1, -1)
    BLOCK_SIZE = 128
    grid = (N, 1, 1)
    output = torch.zeros_like(input)
    ## output is returned by the kernel
    ct.launch(torch.cuda.current_stream(), grid, layer_norm_kernel, (input, output, weight, bias, C, H, W, eps, BLOCK_SIZE))
    return output.view(N, C, H, W)


def pytorch_reference(model, input):
    return model(input)


if __name__ == "__main__":
    torch.manual_seed(42)
    N, C, H, W = 4, 8, 32, 16
    dtype = torch.float16

    assert dtype == torch.float16, "Only float16 is supported"

    class LayerNorm(torch.nn.Module):
        def __init__(self, normalized_shape):
            super(LayerNorm, self).__init__()
            self.layer_norm = torch.nn.LayerNorm(normalized_shape)

        def forward(self, x):
            return self.layer_norm(x)

    model = LayerNorm((C, H, W)).eval().to(dtype).to("cuda")

    # Create test input
    input_tensor = torch.rand(N, C, H, W, dtype=dtype, device="cuda")
    weight = model.layer_norm.weight.data
    bias = model.layer_norm.bias.data

    eps = 1e-5
    output_cutile = cutile_layernorm(input_tensor, weight, bias, eps)
    # output_cutile = pytorch_layernorm_manual(input_tensor, weight, bias, eps)

    # Test against PyTorch's built-in LayerNorm
    output_pytorch = pytorch_reference(model, input_tensor)

    # Validate results
    if torch.allclose(output_cutile, output_pytorch, atol=1e-2, rtol=1e-2):
        print("Test passed!")
    else:
        print("Test failed!")
        abs_diff = torch.abs(output_cutile - output_pytorch)
        print(f"Max absolute difference: {torch.max(abs_diff).item():.6f}")
        print(f"Mean absolute difference: {torch.mean(abs_diff).item():.6f}")
