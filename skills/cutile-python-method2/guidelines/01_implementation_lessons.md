# cuTile - Implementation Lessons

## Important lessons for cuTile

You will see a lot of lessons with code examples below.

Lesson 1: cuTile is a tile-based programming model, so you need to use the tile index to access the data.
- Wrong code: `a = ct.load(A, index=(bid_m * BLOCK_M, k_tile), shape=(BLOCK_M, BLOCK_K))`
- Correct code: `a = ct.load(A, index=(bid_m, k_tile), shape=(BLOCK_M, BLOCK_K))`

Lesson 2: The same thing applies to the store operation.
- Wrong code: `ct.store(output, index=(bid_m * BLOCK_M, bid_n * BLOCK_N), tile=acc)`
- Correct code: `ct.store(output, index=(bid_m, bid_n), tile=acc)`

Lesson 3: When accumulator is used, you need to use a promoted data type.
```python
# original dtype is float16
sum = ct.full(shape, 0, dtype=np.float32)
# do some computation
sum = ct.astype(sum, np.float16)  # change the data type of sum back to float16
```

Lesson 4: Use `ct.num_tiles` to get the number of tiles as `math.ceil` is not allowed in a cuTile kernel.
Note that the given tile shape must be the same as the shape of the input tensor.
- Wrong code: `num_tiles = ct.num_tiles(A, axis=1, shape=(tk,))` when `A` is a 2D tensor
- Correct code: `num_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))` when `A` is a 2D tensor

Lesson 5: Use `ct.astype` to change the data type of the accumulator.
```python
# original dtype is float16
sum = ct.full(shape, 0, dtype=np.float32)
# do some computation
sum = ct.astype(sum, np.float16)  # change the data type of sum back to float16
```

Lesson 6: `ct.astype` is only for tile or scalar data type, not for constant data type.
- Wrong code: `ct.astype(1.0, np.float32)`
- Correct code: `tx = ct.astype(tx, np.float32)`


Lesson 7: `ct.ones` is not allowed in a cuTile kernel, but `ct.full` is allowed.
```python
# Wrong code
ones = ct.ones(shape, dtype=np.float32)
# Correct code
ones = ct.full(shape, 1, dtype=np.float32)
```

Lesson 8: Use `ct` namespace to get the data type of the tensor in cuTile kernels.
- Such as `ct.float32`, `ct.float16`, `ct.int32` and etc.

Lesson 9: Since cuTile only supports 2D and 3D matrix multiplication, you need to use reshape to convert the tensor to 2D or 3D.
```python
# Input tensor A is 4D (B, M, N, K), but cuTile only supports 2D and 3D matrix multiplication
tx = ct.load(A, index=(bid_b, bid_m, bid_n, bid_k), shape=(BLOCK_B, BLOCK_M, BLOCK_N, BLOCK_K))
# Reshape the tensor to 3D (B * M, N, K)
tx = ct.reshape(tx, (B * M, N, K))
# Input tensor B is 4D (B, M, K, N), but cuTile only supports 2D and 3D matrix multiplication
ty = ct.load(B, index=(bid_b, bid_m, bid_k, bid_n), shape=(BLOCK_B, BLOCK_M, BLOCK_K, BLOCK_N))
# Reshape the tensor to 3D (B * M, K, N)
ty = ct.reshape(ty, (B * M, K, N))
# Do matrix multiplication (B * M, N, K) * (B * M, K, N) -> (B * M, N, N)
tz = ct.matmul(tx, ty)
# Reshape the result back to 4D (B, M, N, N)
tz = ct.reshape(tz, (B, M, N, K))
# Store the result
ct.store(C, index=(bid_b, bid_m, bid_n, bid_n), tile=tz)
```

Lesson 10: Using a loop for tile accumulation is supported when memory is a problem, such as the case of matrix multiplication.  Please note that the loop should iterates over the reduction dimension for this case.
```python
# Matrix multiplication example with 3D tensors:
#   Input: A (B, M, K), B (B, K, N)
#   Output: C (B, M, N)

# Get the number of tiles along the axis 2 of the input tensor A
num_tiles = ct.num_tiles(A, axis=2, shape=(tb, tm, tk))
# Need to accumulate the result, using float32 as the accumulator type
acc = ct.full(shape=(BLOCK_B, BLOCK_M, BLOCK_N), value=0, dtype=np.float32)
for k in range(num_tiles):
    # Create a tile from the input tensor A, the shape of the tile is (BLOCK_B, BLOCK_M, BLOCK_K)
    tx = ct.load(A, index=(bid_b, bid_m, k), shape=(BLOCK_B, BLOCK_M, BLOCK_K))
    # Create a tile from the input tensor B, the shape of the tile is (BLOCK_B, BLOCK_K, BLOCK_N)
    ty = ct.load(B, index=(bid_b, k, bid_n), shape=(BLOCK_B, BLOCK_K, BLOCK_N))
    # Do tile matrix multiplication (B, M, K) * (B, K, N) -> (B, M, N)
    acc = ct.mma(tx, ty, acc)
# Cast type to the output tensor C
acc = ct.astype(acc, C.dtype)
# Store the result
ct.store(C, index=(bid_b, bid_m, bid_n), tile=acc)
```

Lesson 11: Constants in cuTile cannot be initialized with its type.
- Wrong code: `x:ct.Constant[int] = ct.Constant[int](1)`
- Correct code: `x = 1` (It is optional to omit the type annotation)


Lesson 12: When the problem size is large, you need to estimate the number of tiles and the block size.
The maximum total number of threads in a grid is 65535.
If the problem size is large, you need to estimate the number of tiles and the
block size to ensure the total number of threads is less than 65535.


## Tile size restriction

Each dimension of a tile must be a power of 2 (i.e., 2^n) when using `ct.load` and `ct.store` to load and store the tile.
If the requested tile shape contains any dimension that is **not** a power of 2, cuTile will return an error.
Thus, we need to pass a new parameter for the next larger power-of-2 tile size
and the excess elements are padded with zeros (or the specified padding mode, default is `ct.PaddingMode.ZERO`).

Example:

```python
def next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

@ct.kernel
def kernel(x, SIZE: ct.Constant[int], SIZE_P: ct.Constant[int]):
    bid_0 = ct.bid(0)
    bid_1 = ct.bid(1)
    ## Wrong code: tx = ct.load(x, index=(bid_0, bid_1), shape=(SIZE, SIZE)) ## Not a power of 2
    tx = ct.load(x, index=(bid_0, bid_1), shape=(SIZE_P, SIZE_P))
    ## Do some computation on the tile
    ct.store(x, index=(bid_0, bid_1), tile=...) ## The tile is padded with zeros (default)

size = 10  ## Not a power of 2
size_p = next_power_of_2(size) ## 16, the next larger power of 2
ct.launch(stream, grid, kernel, (x, size, size_p))
```

As can be seen, it is a common practice to pass both the original size and the next larger power of 2 size to the kernel as kernel parameters.
This is because the kernel code does not need to know the original size, but only the next larger power of 2 size.


## Understanding Memory Operations in cuTile

`ct.load` and `ct.store` are fundamental operations for managing data movement in cuTile:

1. `ct.load`:
   - Moves data from global memory to tile registers
   - Cannot be used to move data between tile registers
   - For tile-to-tile operations, use NumPy-style operations like:
     - Reshape: `ct.reshape(tile, new_shape)`
     - Transpose: `ct.transpose(tile, axis0, axis1)`
     - Indexing: `tile[:, :, 0:5]`
2. `ct.store`:
   - Moves data from tile registers back to global memory
   - Is the inverse operation of `ct.load`
   - Must match the data type of the destination tensor

Example: Understand the shape of tile from the shape of the input tensor
```python
# In ct.load, the parameter `index` defines the starting point of the tile,
#             the parameter `shape` defines the shape of the tile.
# The same also applies to ct.store

# Create a tile from the input tensor A, the shape of the tile is (BLOCK_B, BLOCK_M)
tx = ct.load(A, index=(bid_b, bid_m), shape=(BLOCK_B, BLOCK_M))
# This creates the same tile shape as tx, but the index is (0, bid_m)
ty = ct.load(A, index=(0, bid_m), shape=(BLOCK_B, BLOCK_M))
```

## Kernel fusion in cuTile

Kernel fusion is essential in cuTile to maximize performance and minimize memory traffic.  Key principles for effective kernel fusion:
1. Maintain consistent tile indices across fused operations
2. Analyze input tensor shapes and block sizes to ensure compatible tile indices
3. Maximize the number of operations within a single kernel
4. Consider memory access patterns when fusing operations

Common kernel fusion patterns:
1. Element-wise operations:
   - Addition, multiplication, or other element-wise operations between tensors
   - Example: A + B where A and B share the same tile indices
2. Matrix multiplication with activation:
   - Fuse matrix multiplication with element-wise operations
   - Example: ReLU(matmul(A, B)) where A and B maintain consistent tile indices
3. Chained matrix operations:
   - Fuse multiple matrix operations that share input tensors
   - Example: matmul(matmul(A, B), C) where A's tile indices are preserved

Best practices:
- Always verify tile index compatibility before fusion
- Use the same block sizes for operations that will be fused
- Consider memory bandwidth when deciding which operations to fuse
- Profile performance to validate fusion benefits

## Default rules when user does not specify

1. **Default Data Type**: If tensor types are not specified, use `torch.float16` as the default data type for optimal GPU memory usage and performance.

2. **Default Tolerance Values**: If numerical comparison tolerance is not specified, use the following defaults based on data type:
   - `torch.float32`: `atol=1e-3, rtol=1e-3`
   - `torch.float16` and `torch.bfloat16`: `atol=1e-2, rtol=1e-2`
   - These tolerances account for the reduced precision of half-precision formats while ensuring meaningful validation.

3. **Default Tensor Shapes**: If tensor shapes are not specified, generate suitable shapes where:
   - Each dimension is a power of 2 (e.g., 32, 64, 128, 256)
   - Consider GPU memory constraints and typical use cases
   - For higher dimensions: ensure total elements remain reasonable for testing


Lesson 13: `ct.load(..., order='F')` does NOT transpose the tile. It compiles but produces wrong shapes or results. To transpose a 2D tile (e.g., for `ct.mma`), load it normally and then explicitly transpose:
```python
# WRONG — order='F' does not perform a real transpose:
w = ct.load(A, index=(bid_n, k), shape=(BLOCK_N, BLOCK_K), order='F')

# CORRECT — load then explicitly transpose:
w = ct.load(A, index=(bid_n, k), shape=(BLOCK_N, BLOCK_K))
w_t = ct.transpose(w)   # → (BLOCK_K, BLOCK_N)
# Alternative: ct.permute(w, (1, 0))
```
Never use `order='F'` as a substitute for an explicit transpose.

Lesson 14: Boolean tile arithmetic is NOT supported. Always cast boolean comparison results to int32 before multiplying, or use `ct.where` with a boolean mask:
```python
# WRONG — bool * bool causes a compilation error:
valid = (idx >= 0) * (idx < SIZE)

# CORRECT — cast each comparison to int32 first:
valid = ct.astype(idx >= 0, ct.int32) * ct.astype(idx < SIZE, ct.int32)

# Alternative — use ct.where with a boolean mask:
valid_mask = (idx >= 0) & (idx < SIZE)
result = ct.where(valid_mask, tile, zero_tile)
```

Lesson 15: The tile passed to `ct.store` must have the same rank as the index tuple, which must equal the destination tensor's rank. When a reduction produces a tile of higher rank than needed, use `ct.reshape` to match before storing. Also prefer `keepdims=True` in reductions to keep track of rank:
```python
# Example: 4D input reduced along two axes, stored to a 3D tensor
max_val = ct.max(x, axis=(2, 3), keepdims=True)   # shape (1,1,1,1) — 4D
ct.store(y, index=(bid_a, bid_b, bid_c), tile=max_val)  # WRONG — index is 3D

# CORRECT — reshape tile rank to match:
ct.store(y, index=(bid_a, bid_b, bid_c), tile=ct.reshape(max_val, (1, 1, 1)))
```

Lesson 16: `ct.gather` with out-of-bounds indices does not automatically return zero — it reads garbage memory, which produces NaN. This is a risk whenever BLOCK_SIZE > actual tensor dimension. Fix with two steps:
```python
# 1. Clamp indices to the valid range before gathering:
idx = ct.minimum(idx, DIM - 1)
tx = ct.gather(input, (..., idx, ...), padding_value=0.0)

# 2. Use ct.where (not tile * mask) to zero padded positions:
#    NaN * 0 = NaN, but ct.where(False, NaN, 0) = 0
tx = ct.where(valid_mask, tx, zero_tile)
```

Lesson 17: In cuTile, every variable has a single inferred static type (shape + dtype). If the same variable name is used for tiles of different shapes in different loop bodies, the compiler raises a type conflict. Use unique variable names for tiles that have different shapes:
```python
# WRONG — same name 'tmp' used for tiles of different shapes:
for k in range(n_a):
    tmp = ct.load(A, ..., shape=(BLOCK_K1, BLOCK_N))
for k in range(n_b):
    tmp = ct.load(B, ..., shape=(BLOCK_K2, BLOCK_N))  # CONFLICT

# CORRECT — unique names per loop:
for k in range(n_a):
    tmp_a = ct.load(A, ..., shape=(BLOCK_K1, BLOCK_N))
for k in range(n_b):
    tmp_b = ct.load(B, ..., shape=(BLOCK_K2, BLOCK_N))
```

Lesson 18: All scalar float/int kernel parameters must be annotated as `ct.Constant[float]` or `ct.Constant[int]`. Un-annotated scalars are not recognized by the cuTile type system:
```python
# WRONG:
def kernel(x, eps, scale):  # eps and scale are untyped

# CORRECT:
def kernel(x, eps: ct.Constant[float], scale: ct.Constant[float]):
```

Lesson 19: Python-style slice syntax (`tile[0]`, `tile[:, :, 3:4]`) is NOT supported inside cuTile kernels. Use scalar `ct.load(tensor, index=(...), shape=())` for element access, or restructure using `ct.arange` and masking.

## Reference implementation

When working with user-provided reference implementations, follow these guidelines:

1. **Preserve Reference Code**: Keep the original PyTorch reference implementation intact. Only remove code that is clearly redundant or unnecessary.

2. **Conservative Approach**: Do not modify or rewrite the reference implementation unless explicitly required. The reference serves as the ground truth for correctness validation.

3. **Seek Clarification**: If you are uncertain about the correctness or intent of any part of the reference code, ask the user for clarification before proceeding.

4. **Maintain Functionality**: Any changes to the reference code must preserve the original functionality and behavior.


## Special implementations in unit tests

When implementing complex algorithms in cuTile, you will encounter specialized implementations like flash attention. These optimizations are designed to improve memory efficiency and computational performance.

To successfully implement these algorithms:

1. Analyze the PyTorch implementation:
   - Understand the mathematical operations and data flow
   - Identify key computational patterns and memory access patterns
   - Note any special optimizations or constraints

2. Study relevant cuTile examples in unit tests in the next section:
   - Review unit tests for similar operations, as this is critical since existing examples often provide the exact patterns you need
   - Examine how tile operations are structured, paying special attention to how blocks and indices are handled
   - Understand how memory access is optimized by looking for patterns in load/store operations that you can reuse
   - Copy and adapt working patterns from examples rather than reinventing the wheel when proven solutions exist

3. Implement the cuTile version:
   - Map PyTorch operations to equivalent cuTile primitives
   - Apply kernel fusion where appropriate
   - Ensure proper tile indexing and memory management
   - Validate against the PyTorch reference implementation

This systematic approach ensures accurate and efficient translation from PyTorch to cuTile.
