---
name: cutile-kernel-patterns
description: Core cuTile kernel authoring patterns including grid indexing, tile creation, TMA loads, compile-time constants, and accumulator initialization used across attention, matmul, and element-wise kernels.
category: coding
---

## cuTile Kernel Patterns

- **Grid indexing**: Use `ct.bid(axis)` to map thread blocks — axis 0/1/2 for M/N/batch or query-tile/batch-head/split dimensions.
- **Tile-local indices**: `ct.arange(TILE_SIZE, dtype=ct.int32)` creates index vectors; reshape with `[:, None]` / `[None, :]` for 2D broadcasting.
- **TMA loads**: `ct.load(tensor, index=(...), shape=(...), order=(...), allow_tma=True)` for hardware TMA-backed global→shared transfers. For 1D, use `ct.gather(x, offsets, padding_value=0)` instead.
- **Compile-time constants**: Annotate with `ct.Constant[int]` / `ct.Constant[bool]` (e.g., `TILE_D`, `CAUSAL`, `EVEN_K`) to enable static tiling and control-flow optimization.
- **Accumulators**: Initialize with `ct.full((tm, tn), 0.0, dtype=ct.float32)` for MMA or softmax; column vectors `ct.full((TILE_M, 1), 0.0, ...)` for online softmax (m_i, l_i).
- **MMA**: `ct.mma(a, b, acc=sum)` maps to tensor cores; load 3D tiles then `ct.reshape` to 2D before MMA.
- **K-dim loops**: Iterate via `ct.num_tiles(A, axis=K, shape=tile_shape)` for reduction.
- **Scalar extraction**: Load shape `(1,)` tile then `.item()` for runtime-dynamic values (e.g., sequence positions).
- **Kernel structure**: Separate reusable `fmha_kernel_impl` device function (with explicit `bid_x`, `bid_y`) from `@ct.kernel()` entry point for composability.
