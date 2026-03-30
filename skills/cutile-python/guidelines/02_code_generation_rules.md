# cuTile - Code Generation Rules

## Additional rules learned from the sample tests

Please carefully review and apply each of the following rules when generating cuTile code. These rules are critical for producing correct and functional cuTile kernels.

Rule 1: No `ct.sign` in cuTile kernels. Use `ct.where` to replace it.
```python
# Wrong code
signed_tx = ct.sign(tx)
# Correct code
signed_tx = ct.where(tx > 0, 1, 0) + ct.where(tx < 0, -1, 0) # Use ct.where to replace ct.sign
signed_tx = ct.astype(signed_tx, tx.dtype) # Use ct.astype to replace ct.sign
```

Rule 2: Both `ct.abs(x)` and `abs(x)` are valid in cuTile kernels. `ct.abs` was added in v1.1.0.
```python
# Both are correct
abs_x1 = ct.abs(x1)
abs_x1 = abs(x1)
```

Rule 3: No `ct.neg` in cuTile kernels. Use `-x` to replace it.
```python
# Wrong code
neg_x1 = ct.neg(x1)
# Correct code
neg_x1 = -x1
```

Rule 4: When loading scalar values in cuTile kernels, use `shape=()` for 0D tile (scalar) loads.
```python
# Loading a scalar value from a 1D array as a 0D tile (scalar)
# The index matches the array's rank, shape=() indicates scalar output
tx = ct.load(x, index=(0,), shape=())  # x is 1D array, loads element as scalar

# Loading a scalar from a 3D array as a 0D tile
tx = ct.load(array3d, index=(0, 0, 0), shape=())  # Valid scalar load

# Single-element tiles are valid for scalar broadcasting patterns
value = ct.load(input, index=(bid_x, bid_y), shape=(1, 1))  # Valid for broadcasting

# Note: When shape=(), the index tuple length must match the SOURCE ARRAY's
# dimensionality, not the shape tuple's length.
```

Rule 5: No square in cuTile kernels. Use `x * x` to replace it.
```python
# Wrong code 1
sqr_x = ct.sqr(x)
# Wrong code 2
sqr_x = ct.square(x)
# Correct code
sqr_x = x * x
```

Rule 6: cuTile kernel grid must be a tuple of integers with no more than 3 elements.
```python
# Wrong code 1
grid = 1
# Wrong code 2
grid = (1, 2, 3, 4)
# Wrong code 3
grid = [1] # a list is not a tuple, expect a tuple (1,)
# Correct code 1
grid = (1,)
# Correct code 2
grid = (1, 2)
# Correct code 3
grid = (1, 2, 3)
```

Rule 7: When looping over an axis, use block ids instead of the loop index.
```python
# Wrong code
for i in range(0, 16, BLOCK):
    tx = ct.load(x, index=(i,), shape=(BLOCK,))
# Correct code
for i in range(0, 16, BLOCK):
    block_id = i // BLOCK
    tx = ct.load(x, index=(block_id,), shape=(BLOCK,))
```

Rule 8: Since there is no `ct.flip` in cuTile, you need to implement it yourself. Here is an example of how to implement `ct.flip` in cuTile using a loop with the input 2D tensor.
```python
@ct.kernel
def flip(input, output, dim_1_size: ct.Constant[int]):
    # input shape: (batch, dim_size)
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    value = ct.load(input, (bid_x, dim_1_size - 1 - bid_y), shape=(1, 1))
    ct.store(output, index=(bid_x, bid_y), tile=value)
```

Rule 9: No need additional synchronization after kernel launch in cuTile.

```python
# Wrong code
ct.launch(stream, grid, kernel, kernel_args)
torch.cuda.synchronize()

# Correct code (No need additional synchronization after kernel launch in cuTile)
ct.launch(stream, grid, kernel, kernel_args)
```

Rule 10: Refrain from checking the boundary for out-of-bounds in cuTile kernels.

cuTile automatically handles out-of-bounds accesses with well-defined default values, eliminating the need for manual boundary checks. This applies to:

- **Tile loads/stores**: Out-of-range indices return zeros (or other appropriate defaults) instead of causing errors
- **Block indices (`ct.bid`)**: These are guaranteed to be within valid ranges based on the grid dimensions you specify
- **Memory operations**: `ct.load()` and `ct.store()` safely handle edge cases without explicit bounds checking

**Key insight**: Unlike CUDA C/C++ where out-of-bounds accesses can cause *undefined behavior, cuTile provides safe defaults.
This simplifies kernel code *significantly - you can focus on the core computation logic rather than *defensive programming against boundary conditions.


Rule 11: Prefer tiled based programming over loop based programming in cuTile.

- Total grid size 1 should avoided unless the problem size is small.
- When the problem size is large, you need to estimate the number of tiles and the block size.

Rule 12: Use `rand` instead of `randn` to generate random numbers in cuTile kernels.

- This is because `randn` generates random numbers from a normal distribution, which may not be suitable for validating the cuTile kernel.


## Tolerance for numerical validation

Please note that the tolerance for numerical validation is critical for accurate testing. The tolerance values must be carefully balanced:

- **Too strict**: Tests may fail for correct implementations due to minor numerical differences from floating-point precision limitations
- **Too loose**: Tests may pass for incorrect implementations, missing actual bugs or errors

**Recommended tolerance values:**
- **float32**: `rtol=1e-3, atol=1e-3`
- **float16/bfloat16**: `rtol=1e-2, atol=1e-2`

These values account for the inherent precision limitations of each data type while maintaining sufficient sensitivity to detect implementation errors.


Rule 13: Never use `ct.tfloat32`. Use `float16` inputs with `float32` accumulators.

**The default compute pattern for matmul is: load inputs as `float16`, accumulate in `float32`.** Do not use `ct.tfloat32` — it causes validation failures (~0.1 max absolute error) and is unnecessary when inputs are already float16.

If the input tensor arrives as float32, cast it to float16 on load:

```python
# CORRECT: float16 inputs, float32 accumulator
acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
for k in range(num_k):
    a = ct.astype(ct.load(A, index=(bid_m, k), shape=(BLOCK_M, BLOCK_K)), ct.float16)
    b = ct.astype(ct.load(B, index=(k, bid_n), shape=(BLOCK_K, BLOCK_N)), ct.float16)
    acc = ct.mma(a, b, acc)
out = ct.astype(acc, output.dtype)

# WRONG: casting to tfloat32 — causes ~0.1 precision error, do not copy this pattern
a = ct.astype(ct.load(A, ...), ct.tfloat32)  # DO NOT DO THIS
```

Note: TileGym examples sometimes cast float32 inputs to `ct.tfloat32` for throughput. **Do not follow that pattern** — it breaks validation. Always use float16 inputs.

Rule 14: `ct.mma` requires x and y to have the same dtype (unless they are int8/uint8). Cast both inputs to the same type before calling `ct.mma`.

```python
# WRONG: x is float32, y is float16 — TileTypeError in v1.2.0+
p = ct.exp(qk)         # float32
v = ct.load(V, ...)    # float16
o = ct.mma(p, v, o)

# CORRECT: cast y to match x
p = ct.exp(qk)                    # float32
v = ct.load(V, ...)               # float16
v = ct.astype(v, np.float32)      # now float32 — matches p
o = ct.mma(p, v, o)
```

Rule 15: `ct.cumsum` works correctly on both 1D `(L,)` with `axis=0` and 2D `(1, L)` with `axis=1`. The 2D form is safer and more idiomatic:

```python
# Both are correct, but 2D form is preferred
ct.cumsum(tile_1d, axis=0)            # tile_1d shape: (L,)
ct.cumsum(tile_2d, axis=1)            # tile_2d shape: (1, L)  ← preferred
```

Rule 16: When debugging large numerical errors, always check BOTH absolute and relative errors before concluding the kernel is wrong.

A large `max_diff` can be misleading — it may reflect float32 noise on large-valued outputs rather than an algorithmic bug. Before investigating the kernel, compute:

```python
abs_diff = (actual - expected).abs()
rel_diff = abs_diff / (expected.abs() + 1e-8)
print(f"abs_max={abs_diff.max():.3e}, rel_max={rel_diff.max():.3e}, rel_mean={rel_diff.mean():.3e}")
```

If `rel_mean` is small (e.g., < 1e-4) but `abs_max` is large, the kernel is likely correct — the large absolute error comes from float32's limited mantissa on large-valued outputs. For example, if outputs reach 1e10, one ULP ≈ 1e10 × 2⁻²³ ≈ 1200 absolute error, which is expected and not a bug.

Rule 17: Numerical test inputs should reflect the physical or mathematical constraints of the algorithm.

Unconstrained random inputs can create ill-conditioned problems where outputs have enormous magnitudes, causing catastrophic cancellation in the final result. This is not a cuTile bug — it is a property of the algorithm under pathological inputs.

Best practice: use two-tier validation:

```python
# Tier 1: Shape and dtype check with arbitrary random input
shape_ok = actual.shape == expected.shape and actual.dtype == expected.dtype

# Tier 2: Numerical accuracy with constrained input that matches real usage
# e.g., if the algorithm requires inputs to be bounded or have a specific sign,
# construct test inputs that satisfy those constraints
x_constrained = construct_valid_input(...)
is_close = torch.allclose(actual_constrained, expected_constrained, atol=1e-3, rtol=1e-3)
```

When the algorithm has known input constraints (e.g., values must be negative, normalized, or bounded), use inputs that satisfy those constraints for numerical validation.

Rule 18: Never fall back to PyTorch ops (F.conv2d, F.conv3d, F.conv_transpose2d, F.conv_transpose3d, F.linear, torch.matmul, etc.) because the grid "would be too large" or the kernel "would be complex." These are always solvable with spatial tiling. This applies equally to transposed convolutions and 3D convolutions — they are not special cases.

Large spatial outputs are not a reason to use F.conv2d or F.conv3d. Tile the spatial dimension so each block handles BLOCK_HW output positions:

```python
# Wrong reasoning: "grid would be too large → use F.conv2d"
# Correct: tile the spatial dimension
BLOCK_HW = 128
grid = (N, out_channels, ct.cdiv(out_H * out_W, BLOCK_HW))
# each block handles BLOCK_HW output positions
```

Varying kernel parameters across invocations (e.g. different channel counts per layer) are handled with `ct.Constant[int]` and power-of-2 padding — the same kernel template works for all variants.

The rule is simple: **every compute op in the forward pass gets a cuTile kernel**. Grid size and parameter variation are implementation details, not justifications for a PyTorch fallback.

Rule 19: When BN→ReLU→Conv cannot be fused into a single kernel, implement a separate cuTile BN+ReLU kernel — do not fall back to PyTorch for the normalization step.

BN→ReLU→Conv cannot be folded into a single Conv kernel because ReLU breaks the linearity required for BN weight folding. The correct fix is two sequential cuTile kernels, not a PyTorch fallback:

```python
# Wrong: BN+ReLU in PyTorch, only Conv in cuTile
x_bn = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
x_relu = torch.relu(x_bn)   # ← short-circuit
y = launch_conv(x_relu, weight)

# Correct: both steps in cuTile
x_normed = launch_bn_relu(x, gamma, beta, running_mean, running_var)
y = launch_conv(x_normed, weight)
```

The BN+ReLU kernel is straightforward: load a tile, normalize, clamp at zero, store. It is not complex. When a fused BN+ReLU kernel already exists in the file, reuse it — do not reach for `torch.relu` in the compose layer.

Rule 20: `torch.matmul`, `torch.bmm`, and `F.linear` are compute ops. Never label them "no compute", "reshape", or "infrastructure."

These operations are among the most expensive in a forward pass and must be implemented as cuTile kernels. Common misclassifications to avoid:

- Grouping a batched matmul (e.g. `attn @ V`) with data-movement ops like permute/reshape as "no compute" — matrix multiplication is never a reshape.
- Treating `F.linear` projections as "boilerplate that PyTorch handles well" — every linear projection is a GEMM and requires a cuTile kernel.

If you already have a `linear_bias_kernel` or `matmul_kernel` in the file, every `F.linear` / `torch.matmul` / `torch.bmm` call in the composed path must route through it.

## Conversion rules for PyTorch operations

### `torch.min` and `torch.max`

These are reduction operations that compute the minimum/maximum value along a specified axis, not element-wise functions. Therefore, you must use `ct.min`/`ct.max` instead of `ct.minimum`/`ct.maximum`, which perform element-wise comparisons between two tensors.

```python
## PyTorch code
x = torch.min(x, dim=1, keepdim=False)[0]

## cuTile code (correct)
x = ct.min(x, axis=1, keepdims=False)
```
