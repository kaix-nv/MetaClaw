"""Example 1: cumsum / cumprod with blocking"""
import torch
import math
import cuda.tile as ct

@ct.kernel
def kernel_cumsum(
    input, output, BATCH_SIZE_BLOCK: ct.Constant[int], INPUT_SIZE: ct.Constant[int]
):
    bid = ct.bid(0)
    tx = ct.load(input, index=(bid, 0), shape=(BATCH_SIZE_BLOCK, INPUT_SIZE))
    tz = ct.cumsum(tx, axis=1) ## reduce dimension is the last dimension
    ct.store(output, index=(bid, 0), tile=tz)

def test_cumsum(input, output, batch_size, input_size, batch_size_block):
    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(batch_size / batch_size_block),), ## 1D grid
        kernel_cumsum,
        (input, output, batch_size_block, input_size)
    )
    return output

def torch_reference(input):
    return torch.cumsum(input, dim=1)


torch.manual_seed(42)
batch_size = 128  ## non-reduction dimension
input_size = 256  ## reduction dimension (axis=1)
batch_size_block = 32
input = torch.rand(batch_size, input_size, dtype=torch.float16, device="cuda")
output = torch.zeros_like(input, device="cuda")
output = test_cumsum(input, output, batch_size, input_size, batch_size_block)
ref = torch_reference(input)

if torch.allclose(output, ref, atol=1e-2, rtol=1e-2):
    print("✅ Test passed!")
else:
    print("❌ Test failed!")
