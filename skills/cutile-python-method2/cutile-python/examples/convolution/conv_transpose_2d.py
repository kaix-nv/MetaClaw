"""Test 1: This test below implements 2D convolution transpose with bias, dilation, groups, and output_padding."""
import torch
import cuda.tile as ct


# cuTile kernel implementation
@ct.kernel
def conv_transpose_2d_kernel(
    input,
    weights,
    conv_transpose_bias,
    model_bias,
    output,
    out_channels: ct.Constant[int],
    kernel_size_h: ct.Constant[int],
    kernel_size_w: ct.Constant[int],
    kernel_size_h_p: ct.Constant[int],
    kernel_size_w_p: ct.Constant[int],
    stride_h: ct.Constant[int],
    stride_w: ct.Constant[int],
    padding_h: ct.Constant[int],
    padding_w: ct.Constant[int],
    dilation_h: ct.Constant[int],
    dilation_w: ct.Constant[int],
    height: ct.Constant[int],
    width: ct.Constant[int],
    in_channels_per_group: ct.Constant[int],
    in_channels_per_group_p: ct.Constant[int],
    out_channels_per_group: ct.Constant[int],
):
    b = ct.bid(0) // out_channels
    oc = ct.bid(0) % out_channels
    h = ct.bid(1)
    w = ct.bid(2)

    group_id = oc // out_channels_per_group
    oc_in_group = oc % out_channels_per_group
    ic_start = group_id * in_channels_per_group
    ic_end = ic_start + in_channels_per_group

    # For transpose convolution, we need to find input positions that contribute to each output position
    # The base position for input calculation - this is the "left-top" input position that could contribute
    nh_base = h + padding_h
    nw_base = w + padding_w

    # create range for gather: need to expand the range to the full kernel size
    kernel_range_h = ct.arange(kernel_size_h_p, dtype=ct.int32)
    kernel_range_w = ct.arange(kernel_size_w_p, dtype=ct.int32)
    range_ic = ct.arange(in_channels_per_group_p, dtype=ct.int32) + ic_start

    # Mask out invalid kernel positions (when power-of-2 padding creates extra positions)
    mask_kernel_h = kernel_range_h < kernel_size_h
    mask_kernel_w = kernel_range_w < kernel_size_w

    # For transpose conv: input_pos = (output_pos + padding - kernel_pos * dilation) / stride
    # We need to check that this results in integer positions within bounds
    range_h = (nh_base - kernel_range_h * dilation_h) // stride_h
    range_w = (nw_base - kernel_range_w * dilation_w) // stride_w

    mask_ic = (range_ic >= ic_start) & (range_ic < ic_end)

    # For transpose convolution, we need to check:
    # 1. Input positions are within bounds
    # 2. The division results in exact integers (no remainder)
    # 3. Kernel positions are valid (within actual kernel size)
    mask_h_bounds = (range_h >= 0) & (range_h < height)
    mask_w_bounds = (range_w >= 0) & (range_w < width)

    # Check that the kernel position exactly contributes to this output position
    mask_h_exact = ((nh_base - kernel_range_h * dilation_h) % stride_h) == 0
    mask_w_exact = ((nw_base - kernel_range_w * dilation_w) % stride_w) == 0

    mask_h = mask_kernel_h & mask_h_bounds & mask_h_exact
    mask_w = mask_kernel_w & mask_w_bounds & mask_w_exact

    # create mask with outer product
    mask = (
        mask_ic[None, :, None, None]
        & mask_h[None, None, :, None]
        & mask_w[None, None, None, :]
    )

    index_b = ct.full(1, b, dtype=ct.int32)[:, None, None, None]
    index_ic = range_ic[None, :, None, None]
    index_h = range_h[None, None, :, None]
    index_w = range_w[None, None, None, :]
    indices = (index_b, index_ic, index_h, index_w)
    tx = ct.gather(input, indices, padding_value=0) * mask.astype(input.dtype)

    # load weights (in_channels_p, 1, kernel_size_p, kernel_size_p)
    tw = ct.load(
        weights,
        (group_id, oc_in_group, 0, 0),
        shape=(
            in_channels_per_group_p,
            1,
            kernel_size_h_p,
            kernel_size_w_p,
        ),
    )
    # change shape to (1, in_channels_p, kernel_size_h_p, kernel_size_w_p)
    tw = ct.permute(tw, (1, 0, 2, 3))

    sum_val = ct.sum(tx * tw, axis=(1, 2, 3), keepdims=True)
    tb = ct.load(conv_transpose_bias, (oc,), shape=(1,))
    conv_val = sum_val + tb
    model_bias_val = ct.load(model_bias, (oc, 0, 0), shape=(1, 1, 1))
    result = conv_val + model_bias_val
    ct.store(output, (b, oc, h, w), result)


# PyTorch reference implementation
def pytorch_reference(x_0, model):
    return model(x_0)


# Main execution and validation
if __name__ == "__main__":
    torch.manual_seed(42)
    height = 5
    width = 5
    batch_size = 8  # int
    in_channels = 64  # int
    out_channels = 128  # int
    kernel_size = (3, 4)  # int or tuple
    stride = (2, 2)  # int or tuple
    padding = (0, 0)  # int or tuple
    output_padding = (0, 0)  # int or tuple
    groups = 1  # int
    dilation = (1, 1)  # int or tuple

    assert (
        in_channels % groups == 0
    ), f"in_channels ({in_channels}) must be divisible by groups ({groups})"
    assert (
        out_channels % groups == 0
    ), f"out_channels ({out_channels}) must be divisible by groups ({groups})"

    # Define PyTorch model
    class SimpleConvTranspose2D(torch.nn.Module):
        def __init__(self):
            super(SimpleConvTranspose2D, self).__init__()
            self.conv_transpose1 = torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                dilation=dilation,
            )
            self.bias = torch.nn.Parameter(
                torch.rand(out_channels, 1, 1)
            )  # model bias

        def forward(self, x):
            x = self.conv_transpose1(x)
            x = x + self.bias
            return x

    model = SimpleConvTranspose2D().eval().to("cuda")

    # Create input tensor
    input_tensor = torch.rand(
        batch_size,
        in_channels,
        height,
        width,
        dtype=torch.float32,
        device="cuda",
    )

    # Set kernel dimension stride, padding, output_padding, dilation
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    padding_h, padding_w = padding if isinstance(padding, tuple) else (padding, padding)
    output_padding_h, output_padding_w = (
        output_padding
        if isinstance(output_padding, tuple)
        else (output_padding, output_padding)
    )
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, tuple) else (dilation, dilation)
    )
    kernel_size_h, kernel_size_w = (
        kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    )

    # Create output tensor for cuTile
    out_height = (
        (height - 1) * stride_h
        - 2 * padding_h
        + (kernel_size_h - 1) * dilation_h
        + output_padding_h
        + 1
    )
    out_width = (
        (width - 1) * stride_w
        - 2 * padding_w
        + (kernel_size_w - 1) * dilation_w
        + output_padding_w
        + 1
    )

    output_cudatile = torch.empty(
        [batch_size, out_channels, out_height, out_width],
        dtype=torch.float32,
        device="cuda",
    )

    # Get weights from the model
    weights_tensor = (
        model.conv_transpose1.weight.data
    )  ## shape: (in_channels, out_channels, kernel_size_h, kernel_size_w)
    conv_transpose_bias = model.conv_transpose1.bias.data  ## shape: (out_channels,)
    model_bias = model.bias.data  ## shape: (out_channels, 1, 1)

    def next_power_of_2(x: int) -> int:
        return 1 << (x - 1).bit_length()

    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    in_channels_per_group_p = next_power_of_2(in_channels_per_group)

    # Launch cuTile kernel (batach_size, out_channels, out_height * out_width)
    #   Since cuTile only supports upto 3D grid, we need to launch the kernel for each output channel
    grid = (batch_size * out_channels, out_height, out_width)
    kernel_size_h_p = next_power_of_2(kernel_size_h)
    kernel_size_w_p = next_power_of_2(kernel_size_w)
    in_channels_p = next_power_of_2(in_channels)
    ct.launch(torch.cuda.current_stream(), grid, conv_transpose_2d_kernel, (
        input_tensor,
        weights_tensor,
        conv_transpose_bias,
        model_bias,
        output_cudatile,
        out_channels,
        kernel_size_h,
        kernel_size_w,
        kernel_size_h_p,
        kernel_size_w_p,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        height,
        width,
        in_channels_per_group,
        in_channels_per_group_p,
        out_channels_per_group)
    )

    # PyTorch reference execution
    with torch.no_grad():
        ref_output = model(input_tensor)

    # Numerical validation
    assert not torch.isnan(output_cudatile).any(), "cuTile output contains NaN values"
    assert not torch.isinf(output_cudatile).any(), "cuTile output contains Inf values"
    assert (
        output_cudatile.dtype.is_floating_point
    ), f"cuTile output tensor must be floating point, got {output_cudatile.dtype}"
    if not torch.allclose(output_cudatile, ref_output, atol=1e-2, rtol=1e-2):
        print("cuTile output does not match PyTorch reference")
        max_diff = torch.max(torch.abs(output_cudatile - ref_output))
        print(f"Max difference: {max_diff}")
    else:
        print("Test passed!")
