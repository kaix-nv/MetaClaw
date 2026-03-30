# cuTile Code Generation Rules

Follow these rules when generating cuTile code:

## General Guidelines
- Always validate tensor dimensions before operations
- Check API compatibility and correct function signatures
- Ensure proper memory allocation, especially for output tensors
- Verify build configuration requirements


## Memory Error Prevention
1. Always validate tensor dimensions after flatten() operations to ensure they do not exceed cuTile's memory capacity (2^31 elements).

2. Never access tensor elements without explicit bounds checking using tensor.dim(n) for each dimension in 3D convolutions.

3. Always use power-of-two padding dimensions and clamp thread indices to valid tensor ranges using min/max operations.

4. Always calculate global memory indices using the original tensor dimensions rather than flattened indices for multidimensional operations.

5. Ensure kernel launch parameters (blockDim, gridDim) multiplied by thread indices never exceed any tensor dimension.

## Other Prevention
1. Explicitly specify dtype=ct.float32 in all ct.full(), ct.zeros(), ct.ones(), and ct.store() operations when generating results for PyTorch comparison.

2. Never assume default dtypes - always check the reference tensor's dtype and ensure cuTile kernels produce matching precision.

3. Add a final dtype cast operation (astype(ct.float32)) to cuTile results when the reference uses float32 precision, even if intermediate computations use lower precision.

## Runtime Error Prevention
1. Always cast cuTile kernel outputs to match the exact dtype of the PyTorch reference tensor using explicit `.to(torch.float32)` or `.float()` calls before any validation.

2. Never pass PyTorch tensors with `.requires_grad=True` to cuTile kernels; always fetch the tensors with `with torch.no_grad()` to wrap tensor assignment.

3. Ensure all tensors used in cuTile operations are on the same CUDA device by calling `.cuda()` or `.to(device)` before kernel execution.

4. Always use float32 accumulation precision for reduction operations in kernels, even when input/output tensors use float16 or bfloat16 dtypes.

## Compilation Error Prevention

1. Round every dimension in ct.load shape parameters to the nearest power of two (1, 2, 4, 8, 16, 32, 64, 128, etc.).

2. `ct.where(condition, x, y)` requires a **tile** condition (element-wise selection). Use Python `if` for scalar/compile-time constant conditions. Never use `ct.where` when the condition is a scalar bool or integer — that causes an error. Never use Python `if` when branching on a tile value — that causes an error.
   ```python
   # WRONG: ct.where with scalar constant condition
   ct.where(some_int_constant > 0, tile_a, tile_b)
   # CORRECT for scalar: use Python if
   if some_int_constant > 0:
       result = tile_a
   else:
       result = tile_b

   # WRONG: Python if on tile condition
   if tile > 0:   # tile is not a scalar!
       ...
   # CORRECT for tile condition: use ct.where
   result = ct.where(tile > 0, tile_a, tile_b)
   ```

3. Extract scalar values from tiles using ct.reduce() or indexing before using in conditions or operations that expect scalars.

4. Use only cuTile-native operations (ct.floor, ct.ceil, ct.exp, ct.log, ct.sqrt) instead of Python built-ins like math.floor or math.exp.

## Config Error Prevention
1. Always round every dimension in any cuTile shape tuple up to the nearest power of two before using it in ct.load, ct.store, ct.full, or any tile creation function.

2. Verify all literal dimensions are powers of two (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) before including them in any shape parameter.

3. Calculate padded dimensions using `next_pow2 = 2**((original_dim - 1).bit_length())` for any non-power-of-two runtime values.

4. Ensure padding and unpadding logic is included when original tensor dimensions don't match power-of-two tile sizes.

5. Never pass non-power-of-two dimensions (like 79, 81, 108, 119, 720) directly to any cuTile shape parameter - always round up first.

## Dimension Error Prevention
1. Always ensure every dimension in `ct.load`/`ct.store` shape tuples is an exact power of two (1, 2, 4, 8, 16, 32, 64, 128, etc.) before kernel compilation.

2. Ensure the index tuple length in `ct.load`/`ct.store` exactly matches the tensor's rank - use 3-tuple for 3D tensors, 4-tuple for 4D tensors, etc.

3. Ensure all branches of conditional statements return tiles with identical shapes - avoid mixing different tile ranks or dimensions across if/else paths.

4. Remove extraneous singleton dimensions from output tensors to match expected PyTorch tensor shapes (e.g., [4,8,1,1] → [4,8]).

5. The tile passed to `ct.store()` must have the **same rank** as the output array. Storing a lower-rank tile into a higher-rank array causes a compilation failure or timeout. Use `ct.reshape()` to match ranks before storing.
   ```python
   # WRONG: rank mismatch — tile is rank-1, output is rank-4
   result = ct.full((1,), value, dtype=ct.float32)
   ct.store(output, index=(i0, i1, i2, i3), tile=result)  # compilation failure or timeout

   # CORRECT: reshape tile to match output rank
   result = ct.reshape(result, (1, 1, 1, 1))
   ct.store(output, index=(i0, i1, i2, i3), tile=result)
   ```

## Api Misuse Prevention
1. Always verify all tile dimensions are powers of two (1, 2, 4, 8, 16, 32, 64, 128, 256) before using them in `ct.load()` or `ct.store()` shape parameters.

2. Never use Python built-in functions like `round()`, Python scalar methods like `.any()`, `.floor()`, `.numel()`, or `.size()` on cuTile tensors - use only cuTile-native operations.

3. Ensure kernel launch syntax uses tuple format for grid dimensions: `ct.launch(torch.cuda.current_stream(), grid, kernel, (...))` with maximum 4 dimensions, and pass all arguments positionally without keyword arguments.

4. Check cuTile language specification before using any function - do not use `ct.norm()`, `ct.softmax()`, `ct.sigmoid()`, `ct.flip()`, `ct.thread_id`, `ct.empty()`, `ct.tensor` (lowercase), or other non-existent attributes. Implement missing functions from primitives:
   ```python
   # ct.sigmoid does NOT exist — implement from primitives:
   sigmoid = 1.0 / (1.0 + ct.exp(-x))

   # ct.silu does NOT exist — implement from primitives:
   silu = x * (1.0 / (1.0 + ct.exp(-x)))
   ```

5. Match data types exactly: use Python scalars for `ct.full()` fill_value, ensure `ct.load()` first argument is an Array not Tile, and use `ct.Tensor` (capital T) for type annotations.

6. Never call `.astype()` as a method on a cuTile tile. Use `ct.astype(tile, dtype)` instead. Similarly, never call `.any()`, `.floor()`, `.numel()`, `.size()`, or `.astype()` as methods on tiles.
   ```python
   # WRONG: method call on tile
   result = tile.astype(ct.float32)
   # CORRECT:
   result = ct.astype(tile, ct.float32)
   ```

7. Always specify `dtype=` when calling `ct.arange()`. The dtype parameter is required.
   ```python
   # WRONG: missing dtype
   idx = ct.arange(N)
   # CORRECT:
   idx = ct.arange(N, dtype=ct.int32)
   ```
