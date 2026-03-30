import torch
import torch.nn as nn
import cuda.tile as ct

# ============================================================
# Utility
# ============================================================
def next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

# ============================================================
# Kernel 1: gru_gate_matmul
# ============================================================
@ct.kernel
def gru_gate_matmul_kernel(
    x,
    weight,
    bias,
    gates,
    BLOCK_M: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    num_k = ct.num_tiles(x, axis=1, shape=(BLOCK_M, BLOCK_K))
    for k in range(num_k):
        x_tile = ct.load(x, index=(bid_m, k), shape=(BLOCK_M, BLOCK_K))
        w_tile = ct.load(weight, index=(bid_n, k), shape=(BLOCK_N, BLOCK_K))
        w_tile_t = ct.transpose(w_tile)
        acc = ct.mma(x_tile, w_tile_t, acc)
    b_tile = ct.load(bias, index=(bid_n,), shape=(BLOCK_N,))
    b_tile = ct.reshape(b_tile, (1, BLOCK_N))
    b_tile = ct.astype(b_tile, ct.float32)
    acc = acc + b_tile
    acc = ct.astype(acc, gates.dtype)
    ct.store(gates, index=(bid_m, bid_n), tile=acc)

def launch_gru_gate_matmul(x, weight, bias):
    batch_size = x.shape[0]
    out_dim = weight.shape[0]
    gates = torch.empty((batch_size, out_dim), dtype=x.dtype, device="cuda")
    BLOCK_M = next_power_of_2(batch_size)
    BLOCK_K = 64
    BLOCK_N = 64
    grid = (ct.cdiv(batch_size, BLOCK_M), ct.cdiv(out_dim, BLOCK_N))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gru_gate_matmul_kernel,
        (x, weight, bias, gates, BLOCK_M, BLOCK_K, BLOCK_N),
    )
    return gates

# ============================================================
# Kernel 2: gru_cell_elementwise
# ============================================================
@ct.kernel
def gru_cell_elementwise_kernel(
    ih_gates,
    hh_gates,
    h_prev,
    h_new,
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    num_n_tiles = ct.num_tiles(h_prev, axis=1, shape=(BLOCK_M, BLOCK_N))
    ih_r = ct.load(ih_gates, index=(bid_m, bid_n), shape=(BLOCK_M, BLOCK_N))
    ih_z = ct.load(ih_gates, index=(bid_m, bid_n + num_n_tiles), shape=(BLOCK_M, BLOCK_N))
    ih_n = ct.load(ih_gates, index=(bid_m, bid_n + 2 * num_n_tiles), shape=(BLOCK_M, BLOCK_N))
    hh_r = ct.load(hh_gates, index=(bid_m, bid_n), shape=(BLOCK_M, BLOCK_N))
    hh_z = ct.load(hh_gates, index=(bid_m, bid_n + num_n_tiles), shape=(BLOCK_M, BLOCK_N))
    hh_n = ct.load(hh_gates, index=(bid_m, bid_n + 2 * num_n_tiles), shape=(BLOCK_M, BLOCK_N))
    hp = ct.load(h_prev, index=(bid_m, bid_n), shape=(BLOCK_M, BLOCK_N))
    ih_r = ct.astype(ih_r, ct.float32)
    ih_z = ct.astype(ih_z, ct.float32)
    ih_n = ct.astype(ih_n, ct.float32)
    hh_r = ct.astype(hh_r, ct.float32)
    hh_z = ct.astype(hh_z, ct.float32)
    hh_n = ct.astype(hh_n, ct.float32)
    hp = ct.astype(hp, ct.float32)
    r = ct.sigmoid(ih_r + hh_r)
    z = ct.sigmoid(ih_z + hh_z)
    n = ct.tanh(ih_n + r * hh_n)
    result = (1.0 - z) * n + z * hp
    result = ct.astype(result, h_new.dtype)
    ct.store(h_new, index=(bid_m, bid_n), tile=result)

def launch_gru_cell_elementwise(ih_gates, hh_gates, h_prev):
    batch_size = h_prev.shape[0]
    hidden_size = h_prev.shape[1]
    h_new = torch.empty((batch_size, hidden_size), dtype=h_prev.dtype, device="cuda")
    BLOCK_M = next_power_of_2(batch_size)
    BLOCK_N = next_power_of_2(hidden_size)
    grid = (ct.cdiv(batch_size, BLOCK_M), ct.cdiv(hidden_size, BLOCK_N))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gru_cell_elementwise_kernel,
        (ih_gates, hh_gates, h_prev, h_new, BLOCK_M, BLOCK_N),
    )
    return h_new

# ============================================================
# Kernel 3: concat_directions
# ============================================================
@ct.kernel
def concat_directions_kernel(
    output_fwd,
    output_bwd,
    output_concat,
    BLOCK_B: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    bid_t = ct.bid(0)
    bid_b = ct.bid(1)
    fwd_tile = ct.load(output_fwd, index=(bid_t, bid_b, 0), shape=(1, BLOCK_B, BLOCK_H))
    bwd_tile = ct.load(output_bwd, index=(bid_t, bid_b, 0), shape=(1, BLOCK_B, BLOCK_H))
    ct.store(output_concat, index=(bid_t, bid_b, 0), tile=fwd_tile)
    ct.store(output_concat, index=(bid_t, bid_b, 1), tile=bwd_tile)

def launch_concat_directions(output_fwd, output_bwd):
    seq_len, batch_size, hidden_size = output_fwd.shape
    output_concat = torch.empty(
        (seq_len, batch_size, 2 * hidden_size), dtype=output_fwd.dtype, device="cuda"
    )
    BLOCK_B = 1
    BLOCK_H = hidden_size
    grid = (seq_len, batch_size)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        concat_directions_kernel,
        (output_fwd, output_bwd, output_concat, BLOCK_B, BLOCK_H),
    )
    return output_concat

# ============================================================
# Composed Function
# ============================================================
def composed_bidirectional_gru(x, gru_model, h0):
    """
    Complete implementation of a 6-layer bidirectional GRU using cuTile kernels.
    Equivalent to nn.GRU(bidirectional=True).

    Args:
        x: input tensor of shape (seq_len, batch_size, input_size), float16
        gru_model: a PyTorch nn.GRU module to extract weights from
        h0: initial hidden state of shape (num_layers * 2, batch_size, hidden_size), float16

    Returns:
        output: tensor of shape (seq_len, batch_size, hidden_size * 2), float16
    """
    num_layers = gru_model.num_layers
    seq_len = x.shape[0]
    batch_size = x.shape[1]
    hidden_size = gru_model.hidden_size

    layer_input = x  # (seq_len, batch_size, input_size) for layer 0, then (seq_len, batch_size, 2*hidden_size)

    for layer in range(num_layers):
        # Extract weights for forward direction
        w_ih_fwd = getattr(gru_model, f"weight_ih_l{layer}")
        w_hh_fwd = getattr(gru_model, f"weight_hh_l{layer}")
        b_ih_fwd = getattr(gru_model, f"bias_ih_l{layer}")
        b_hh_fwd = getattr(gru_model, f"bias_hh_l{layer}")

        # Extract weights for backward direction
        w_ih_bwd = getattr(gru_model, f"weight_ih_l{layer}_reverse")
        w_hh_bwd = getattr(gru_model, f"weight_hh_l{layer}_reverse")
        b_ih_bwd = getattr(gru_model, f"bias_ih_l{layer}_reverse")
        b_hh_bwd = getattr(gru_model, f"bias_hh_l{layer}_reverse")

        # Initial hidden states for this layer
        # h0 layout: [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...]
        h_fwd = h0[layer * 2].clone()      # (batch_size, hidden_size)
        h_bwd = h0[layer * 2 + 1].clone()  # (batch_size, hidden_size)

        # Forward direction: t = 0 .. seq_len-1
        output_fwd = torch.empty(
            (seq_len, batch_size, hidden_size), dtype=x.dtype, device="cuda"
        )
        h_prev = h_fwd
        for t in range(seq_len):
            x_t = layer_input[t]  # (batch_size, feature_dim)
            ih_gates = launch_gru_gate_matmul(x_t, w_ih_fwd, b_ih_fwd)
            hh_gates = launch_gru_gate_matmul(h_prev, w_hh_fwd, b_hh_fwd)
            h_prev = launch_gru_cell_elementwise(ih_gates, hh_gates, h_prev)
            output_fwd[t] = h_prev

        # Backward direction: t = seq_len-1 .. 0
        output_bwd = torch.empty(
            (seq_len, batch_size, hidden_size), dtype=x.dtype, device="cuda"
        )
        h_prev = h_bwd
        for t in range(seq_len - 1, -1, -1):
            x_t = layer_input[t]  # (batch_size, feature_dim)
            ih_gates = launch_gru_gate_matmul(x_t, w_ih_bwd, b_ih_bwd)
            hh_gates = launch_gru_gate_matmul(h_prev, w_hh_bwd, b_hh_bwd)
            h_prev = launch_gru_cell_elementwise(ih_gates, hh_gates, h_prev)
            output_bwd[t] = h_prev

        # Concatenate forward and backward outputs along hidden dimension
        layer_input = launch_concat_directions(output_fwd, output_bwd)

    return layer_input  # (seq_len, batch_size, 2 * hidden_size)

# ============================================================
# Original user-supplied code (copied verbatim, zero modifications)
# ============================================================
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))

    def forward(self, x):
        self.h0 = self.h0.to(x.device)
        output, h_n = self.gru(x, self.h0)
        return output

# ============================================================
# PyTorch Reference
# ============================================================
def pytorch_reference(x, model):
    """Calls the original implementation directly for numerical validation."""
    model.eval()
    with torch.no_grad():
        return model(x)

# ============================================================
# Validation
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    # Create model and move to device with float16
    model = Model(input_size, hidden_size, num_layers, bias=True, batch_first=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Create input
    x = torch.randn(seq_len, batch_size, input_size, device=device, dtype=dtype)

    # Run PyTorch reference
    with torch.no_grad():
        expected = pytorch_reference(x, model)

    # Run composed cuTile implementation
    h0 = model.h0.to(device=device, dtype=dtype)
    actual = composed_bidirectional_gru(x, model.gru, h0)

    # Validate
    is_close = torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)
    max_diff = (actual - expected).abs().max().item()
    if is_close:
        print(f"PASS - max diff: {max_diff}")
    else:
        print(f"FAIL - max diff: {max_diff}")
        print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
        print(f"Actual shape: {actual.shape}, dtype: {actual.dtype}")
