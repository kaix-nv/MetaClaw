"""Test 1: This test below implements 2D convolution with bias, dilation, and groups."""
import torch
import cuda.tile as ct


# cuTile kernel implementation
@ct.kernel
def conv2d_kernel(
    input,
    weights,
    conv_bias,
    model_bias,
    output,
    kernel_size_h: ct.Constant[int],
    kernel_size_h_p: ct.Constant[int],
    kernel_size_w: ct.Constant[int],
    kernel_size_w_p: ct.Constant[int],
    stride_h: ct.Constant[int],
    stride_w: ct.Constant[int],
    padding_h: ct.Constant[int],
    padding_w: ct.Constant[int],
    dilation_h: ct.Constant[int],
    dilation_w: ct.Constant[int],
    height: ct.Constant[int],
    width: ct.Constant[int],
    out_width: ct.Constant[int],
    in_channels_per_group: ct.Constant[int],
    in_channels_per_group_p: ct.Constant[int],
    out_channels_per_group: ct.Constant[int],
):
    b = ct.bid(0)
    oc = ct.bid(1)
    h = ct.bid(2) // out_width
    w = ct.bid(2) % out_width

    group_id = oc // out_channels_per_group
    ic_start = group_id * in_channels_per_group
    ic_end = ic_start + in_channels_per_group

    # Compute the left-top corner of the kernel
    dilated_kernel_size_h = dilation_h * (kernel_size_h - 1) + 1
    dilated_kernel_size_w = dilation_w * (kernel_size_w - 1) + 1
    nh = h * stride_h - padding_h
    nw = w * stride_w - padding_w

    # Create range for gather: need to expand the range to the full kernel size
    kernel_h_range = ct.arange(kernel_size_h_p, dtype=ct.int32)
    kernel_w_range = ct.arange(kernel_size_w_p, dtype=ct.int32)
    range_ic = ct.arange(in_channels_per_group_p, dtype=ct.int32) + ic_start
    range_h = nh + dilation_h * kernel_h_range
    range_w = nw + dilation_w * kernel_w_range
    mask_ic = (range_ic >= ic_start) & (range_ic < ic_end)
    mask_h = (range_h >= 0) & (range_h < min(height, nh + dilated_kernel_size_h))
    mask_w = (range_w >= 0) & (range_w < min(width, nw + dilated_kernel_size_w))

    # Create mask with outer product
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

    # Load weights with load
    tw = ct.load(
        weights,
        (oc, 0, 0, 0),
        shape=(1, in_channels_per_group_p, kernel_size_h_p, kernel_size_w_p),
    )

    sum_val = ct.sum(tx * tw, axis=(1, 2, 3), keepdims=True)
    tb = ct.load(conv_bias, (oc,), shape=(1,))
    # Apply relu
    relu_val = ct.maximum(sum_val + tb, 0)
    model_bias_val = ct.load(model_bias, (oc, 0, 0), shape=(1, 1, 1))
    result = relu_val + model_bias_val
    ct.store(output, (b, oc, h, w), result)


# PyTorch reference implementation
def pytorch_reference(x_0, model):
    return model(x_0)


# Main execution and validation
if __name__ == "__main__":
    torch.manual_seed(42)
    height = 8
    width = 8
    batch_size = 8
    in_channels = 16
    out_channels = 32
    kernel_size = (3, 3)  # int or tuple
    stride = (1, 1)  # int or tuple
    padding = (0, 0)  # int or tuple
    groups = 1  # int
    dilation = (2, 1)  # int or tuple

    assert (
        in_channels % groups == 0
    ), f"in_channels ({in_channels}) must be divisible by groups ({groups})"
    assert (
        out_channels % groups == 0
    ), f"out_channels ({out_channels}) must be divisible by groups ({groups})"

    # Define PyTorch model
    class SimpleConv2D(torch.nn.Module):
        def __init__(self):
            super(SimpleConv2D, self).__init__()
            self.conv1 = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            self.bias = torch.nn.Parameter(
                torch.rand(out_channels, 1, 1)
            )  # model bias
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = (
                x + self.bias
            )  # this model bias is different from the bias of the conv layer
            return x

    model = SimpleConv2D().eval().to("cuda")

    # Create input tensor
    input_tensor = torch.rand(
        batch_size,
        in_channels,
        height,
        width,
        dtype=torch.float32,
        device="cuda",
    )

    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    padding_h, padding_w = padding if isinstance(padding, tuple) else (padding, padding)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, tuple) else (dilation, dilation)
    )
    kernel_size_h, kernel_size_w = (
        kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    )
    # Create output tensor for cuTile
    out_height = (
        height - dilation_h * (kernel_size_h - 1) - 1 + 2 * padding_h
    ) // stride_h + 1
    out_width = (
        width - dilation_w * (kernel_size_w - 1) - 1 + 2 * padding_w
    ) // stride_w + 1
    output_cudatile = torch.empty(
        [batch_size, out_channels, out_height, out_width],
        dtype=torch.float32,
        device="cuda",
    )

    # Get weights from the model
    weights_tensor = (
        model.conv1.weight.data
    )  ## shape: (out_channels, in_channels, kernel_size_h, kernel_size_w)
    conv_bias = model.conv1.bias.data  ## shape: (out_channels,), if bias is True
    model_bias = model.bias.data  ## shape: (out_channels,), if bias is True

    def next_power_of_2(x: int) -> int:
        return 1 << (x - 1).bit_length()

    # Launch cuTile kernel (batach_size, out_channels, out_height * out_width)
    #   Since cuTile only supports upto 3D grid, we need to launch the kernel for each output channel
    grid = (batch_size, out_channels, out_height * out_width)
    kernel_size_h_p = next_power_of_2(kernel_size_h)
    kernel_size_w_p = next_power_of_2(kernel_size_w)
    in_channels_p = next_power_of_2(in_channels)
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    in_channels_per_group_p = next_power_of_2(in_channels_per_group)
    ct.launch(torch.cuda.current_stream(), grid, conv2d_kernel, (
        input_tensor,
        weights_tensor,
        conv_bias,
        model_bias,
        output_cudatile,
        kernel_size_h,
        kernel_size_h_p,
        kernel_size_w,
        kernel_size_w_p,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        height,
        width,
        out_width,
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
    assert torch.allclose(
        output_cudatile, ref_output, atol=1e-2, rtol=1e-2
    ), "cuTile output does not match PyTorch reference"
    print("Test passed!")
