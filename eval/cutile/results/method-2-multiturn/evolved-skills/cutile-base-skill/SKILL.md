---
name: cutile-base-skill
description: Core cuTile API fundamentals: correct module name, kernel launch signature, memory access patterns, block IDs, and key differences from Triton
category: coding
---

## cuTile Core API (NOT Triton)

- **Import**: `from cuda import tile as ct` (NOT `import cutile as ct`)
- **Block/Program ID**: `ct.bid(axis)` for 3D grid axes (NOT `ct.program_id(0)`)
- **Kernel launch**: `ct.launch(torch.cuda.current_stream(), grid, kernel, (args_tuple))` — stream first, grid, kernel, args as tuple (NOT `ct.launch(kernel, grid, *args)`)
- **Memory access**: Use `ct.gather(tensor, indices, ...)` / `ct.scatter(tensor, indices, value, ...)` with `check_bounds=True` or `padding_value` (NOT `ct.load`/`ct.store` with pointer arithmetic and masks)
- **Structured loads**: `ct.load(tensor, (tile_idx_m, tile_idx_k), shape=(TILE_M, TILE_K), order=..., allow_tma=True)` for TMA-based bulk loads
- **Index generation**: `ct.arange(TILE_SIZE, dtype=ct.int32)` (NOT `ct.arange(0, BLOCK_SIZE)`)
- **Type annotations**: Use `ct.Constant[int]`, `ct.Constant[bool]` for compile-time constants
- **No `ct.rand`**: Implement PRNG manually using `ct.bitwise_xor` and bit shifts
- **No `ct.exp`**: Use explicit math ops like `ct.add`, `ct.mul`, `ct.truediv(x, y, flush_to_zero=..., rounding_mode=...)`
- **Accumulators**: `ct.zeros(shape, dtype=...)` instead of manual initialization
