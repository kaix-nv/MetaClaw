---
name: cutile-attention-kernels
description: cuTile patterns specific to fused multi-head attention, flash decode, and attention sink kernels
category: coding
---

## cuTile Attention Kernel Patterns

- **FMHA tiling**: Factor attention into a reusable `_impl` device function that accepts block indices explicitly, enabling the same logic for both standard and attention-sink variants.
- **Online softmax**: Initialize `m_i = ct.full((TILE_M, 1), float('-inf'), ...)` and `l_i = ct.full((TILE_M, 1), 0.0, ...)` accumulators; update per K-tile with running max and sum correction.
- **Causal masking**: Use `ct.Constant[bool]` for `CAUSAL` flag to compile-time branch; generate causal masks from tile-local `ct.arange` offsets.
- **Grouped Query Attention (GQA)**: Multiple Q heads share one KV head via `NUM_Q_HEAD_PER_KV`. Load Q as `(1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM)`, then `ct.reshape()` to 2D before MMA.
- **Split-K parallelism** (flash decode): Split the KV sequence across grid dimension, each CTA computes partial softmax; reduce across splits afterward.
- **Attention sinks**: Load sink count as scalar via `ct.load(Sinks, ...).item()`; handle initial sink tokens separately before processing the main KV range.
- **TMA usage**: Always pass `allow_tma=True` and explicit `order` for Q/K/V loads to enable Tensor Memory Accelerator for efficient global→shared transfers.
- **Occupancy hint**: `@ct.kernel(occupancy=2)` for 2 concurrent CTAs per SM is typical for attention kernels.
