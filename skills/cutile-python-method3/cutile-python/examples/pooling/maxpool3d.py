"""Test 1: This test below implements Max Pooling 3D."""
import torch
import cuda.tile as ct


@ct.kernel
def maxpool3d_kernel(
    input,
    output,
    num_channels: ct.Constant[int],
    depth: ct.Constant[int],
    height: ct.Constant[int],
    width: ct.Constant[int],
    kernel_size_d: ct.Constant[int],
    kernel_size_h: ct.Constant[int],
    kernel_size_w: ct.Constant[int],
    kernel_size_p_d: ct.Constant[int],
    kernel_size_p_h: ct.Constant[int],
    kernel_size_p_w: ct.Constant[int],
    stride_d: ct.Constant[int],
    stride_h: ct.Constant[int],
    stride_w: ct.Constant[int],
    padding_d: ct.Constant[int],
    padding_h: ct.Constant[int],
    padding_w: ct.Constant[int],
    dilation_d: ct.Constant[int],
    dilation_h: ct.Constant[int],
    dilation_w: ct.Constant[int],
):
    bid_n = ct.bid(0) // input.shape[1]  # input.shape[1] = num_channels
    bid_c = ct.bid(0) % input.shape[1]
    bid_d = ct.bid(1)
    bid_h = ct.bid(2) // output.shape[4]  # output.shape[4] = width
    bid_w = ct.bid(2) % output.shape[4]

    # compute the left-top corner of the kernel for pooling with dilation
    dilated_kernel_size_d = dilation_d * (kernel_size_d - 1) + 1
    dilated_kernel_size_h = dilation_h * (kernel_size_h - 1) + 1
    dilated_kernel_size_w = dilation_w * (kernel_size_w - 1) + 1
    d_start = bid_d * stride_d - padding_d
    h_start = bid_h * stride_h - padding_h
    w_start = bid_w * stride_w - padding_w

    range_d = d_start + ct.arange(kernel_size_p_d, dtype=ct.int32) * dilation_d
    range_h = h_start + ct.arange(kernel_size_p_h, dtype=ct.int32) * dilation_h
    range_w = w_start + ct.arange(kernel_size_p_w, dtype=ct.int32) * dilation_w

    mask_d = (range_d >= 0) & (range_d < min(depth, d_start + dilated_kernel_size_d))
    mask_h = (range_h >= 0) & (range_h < min(height, h_start + dilated_kernel_size_h))
    mask_w = (range_w >= 0) & (range_w < min(width, w_start + dilated_kernel_size_w))

    mask = (
        mask_d[None, None, :, None, None]
        & mask_h[None, None, None, :, None]
        & mask_w[None, None, None, None, :]
    )

    index_n = ct.full(1, bid_n, dtype=ct.int32)[:, None, None, None, None]
    index_c = ct.full(1, bid_c, dtype=ct.int32)[None, :, None, None, None]
    index_d = range_d[None, None, :, None, None]
    index_h = range_h[None, None, None, :, None]
    index_w = range_w[None, None, None, None, :]
    indices = (index_n, index_c, index_d, index_h, index_w)
    mask_float = mask.astype(input.dtype)
    tx_raw = ct.gather(input, indices, padding_value=0)
    # Mask out invalid positions with a large negative value (avoids NaN from 0 * -inf)
    tx = tx_raw * mask_float - (1 - mask_float) * 1e30
    max_val = ct.max(tx, axis=(2, 3, 4), keepdims=True)
    result = ct.astype(max_val, output.dtype)
    ct.store(output, (bid_n, bid_c, bid_d, bid_h, bid_w), result)


def pytorch_reference(
    input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
    return torch.nn.functional.max_pool3d(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    # Test parameters
    N, num_channels, depth, height, width = 2, 3, 8, 16, 16
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 2
    ceil_mode = False
    dtype = torch.float32

    # Create test input
    input_tensor = torch.rand(
        N, num_channels, depth, height, width, dtype=dtype, device="cuda"
    )

    # Test manual implementation
    if not ceil_mode:
        depth_output = (
            depth + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        height_output = (
            height + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        width_output = (
            width + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
    else:
        import math
        depth_output = math.ceil(
            (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
        ) + 1
        height_output = math.ceil(
            (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
        ) + 1
        width_output = math.ceil(
            (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
        ) + 1
    assert depth_output > 0, "depth_output must be greater than 0"
    assert height_output > 0, "height_output must be greater than 0"
    assert width_output > 0, "width_output must be greater than 0"

    def next_power_of_2(x: int) -> int:
        return 1 << (x - 1).bit_length()

    kernel_size_p = next_power_of_2(kernel_size)

    output_cutile = torch.zeros(
        N,
        num_channels,
        depth_output,
        height_output,
        width_output,
        dtype=dtype,
        device="cuda",
    )
    ## cuTile kernel only supports up to 3D grid, so we need to flatten some dimensions
    grid = (N * num_channels, depth_output, height_output * width_output)
    ct.launch(torch.cuda.current_stream(), grid, maxpool3d_kernel, (
        input_tensor,
        output_cutile,
        num_channels,
        depth,
        height,
        width,
        kernel_size,
        kernel_size,
        kernel_size,
        kernel_size_p,
        kernel_size_p,
        kernel_size_p,
        stride,
        stride,
        stride,
        padding,
        padding,
        padding,
        dilation,
        dilation,
        dilation)
    )

    # Test against PyTorch's built-in max_pool3d
    output_pytorch = pytorch_reference(
        input_tensor, kernel_size, stride, padding, dilation, ceil_mode
    )

    # Validate results
    if torch.allclose(output_cutile, output_pytorch, atol=1e-6, rtol=1e-6):
        print("Test passed!")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_cutile.shape}")
    else:
        print("Test failed!")
        abs_diff = torch.abs(output_cutile - output_pytorch)
        print(f"Max absolute difference: {torch.max(abs_diff).item():.6f}")
        print(f"Mean absolute difference: {torch.mean(abs_diff).item():.6f}")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Manual output shape: {output_cutile.shape}")
        print(f"PyTorch output shape: {output_pytorch.shape}")
