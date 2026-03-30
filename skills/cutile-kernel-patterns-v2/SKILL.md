---
name: cutile-kernel-patterns
description: cuTile kernel design patterns: tile-based indexing, 2D tile shapes, MMA operations, and multi-dimensional grid strategies
category: coding
---

## cuTile Kernel Design Patterns

- **Tile-based indexing**: Use tile indices with `ct.load(tensor, (tile_m_idx, tile_k_idx), shape=(TILE_M, TILE_K))` — NOT pointer arithmetic like `A + offs_m[:, None] * K`
- **Tile counts**: `ct.num_tiles(tensor, dim, tile_shape)` to compute iteration bounds
- **Always use 2D tile shapes**: Even for vector operations, maintain proper 2D shapes `(TILE_M, TILE_K)` instead of 1D `[D]` with `None` broadcasting
- **Matrix multiply**: Use `ct.mma(a_tile, b_tile, accumulator)` for matrix multiply-accumulate
- **3D grid for attention**: Use `ct.bid(0)`, `ct.bid(1)`, `ct.bid(2)` for (batch, head, tile) instead of flattening into single `ct.program_id(0)` and manually computing indices
- **Masked/irregular access**: Use `ct.gather` with `padding_value` for out-of-bounds safety (e.g., LSE values in split attention)
- **TMA loads**: Add `order` and `allow_tma=True` params for efficient structured tensor memory access
- **Loop over K-tiles**: `for kk in range(num_k_tiles): tile = ct.load(A, (m_idx, kk), shape=...)`
