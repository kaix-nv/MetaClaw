---
name: cutile-attention-kernels
description: cuTile-specific patterns for flash attention variants: prefill FMHA, decode split-K, grouped query attention (GQA), and attention sinks, including online softmax, split-K reduction, and GQA tile layout.
category: coding
---

## cuTile Attention Kernel Patterns

- **Prefill FMHA**: Structure as reusable device fn taking `bid_x`, `bid_y` + `ct.Constant` params (`TILE_M/N/D`, `CAUSAL`). Online softmax uses column-vector accumulators `(TILE_M, 1)` for `m_i` (row max) and `l_i` (row sum).
- **Attention sinks**: Same online softmax pattern; `ct.arange` + reshape creates 2D tile coords for causal masking. Q/K/V loaded via `ct.load(..., shape=(...))` with TMA.
- **Decode (split-K)**: Grid splits KV across `ct.bid(2)`; each split processes `KV_LEN_PER_SPLIT` tokens. Use `ct.load(Start_q, ..., shape=(1,))` then `.item()` for dynamic sequence positions. Partial results written per-split, reduced afterward.
- **GQA decode**: Load Q with shape `(1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM)` and `order=(0,1,2,3)` then `ct.reshape` to 2D. Multiple query heads share one KV head; tile QUERY_GROUP_TILE_SIZE queries together.
- **Common**: `allow_tma=True` on loads for hardware TMA; float32 accumulators for numerical stability; `ct.Constant[bool]` for `CAUSAL` specialization.
- **Composability**: Same `fmha_kernel_impl` can back standalone prefill or fused POD attention by passing different block indices.
