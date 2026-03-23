---
name: cutile-elementwise-and-1d-kernels
description: cuTile patterns for elementwise and 1D operations: gather/scatter instead of load/store, explicit math functions with parameters, and proper type casting
category: coding
---

## cuTile Elementwise & 1D Kernel Patterns

- **Element access**: `ct.gather(tensor, (row_idx, col_offsets), check_bounds=True)` and `ct.scatter(tensor, (row_idx, col_offsets), value, check_bounds=True)`
- **Explicit math ops**: `ct.add(a, b)`, `ct.mul(a, b)`, `ct.truediv(a, b, flush_to_zero=True, rounding_mode='rn')` — no implicit operator overloading assumed
- **Type casting**: `ct.astype(tensor, target_dtype)` for explicit conversions
- **No built-in `ct.exp`/`ct.rand`**: Implement math functions manually or use available primitives
- **Bitwise ops**: `ct.bitwise_xor(a, b)` available for manual PRNG or hash functions
- **Index pattern**: `offsets = ct.arange(TILE_SIZE, dtype=ct.int32)` then `ct.gather(X, (block_idx, offsets))`
- **Bounds checking**: Always use `check_bounds=True` or `padding_value` instead of manual mask computation
- **Constants**: Annotate compile-time params as `ct.Constant[int]` or `ct.Constant[bool]` in kernel signatures
