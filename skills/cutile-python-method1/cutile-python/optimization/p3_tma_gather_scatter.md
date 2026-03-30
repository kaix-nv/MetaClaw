# Section 3: TMA vs Gather/Scatter Selection

> Part of the cuTile Performance Optimization Guide. Originally authored by **Yifei Song** and **Zhengyi Zhang**.

---

**CRITICAL RULE**: TMA requires block-aligned access. Use gather/scatter for arbitrary offsets.

**TMA WORKS** (Block-Aligned Access):
```python
# Example: Regular GEMM, each tile aligned to TILE_M
@ct.kernel
def gemm_kernel(...):
    bid_m, bid_n = ct.bid(0), ct.bid(1)

    # Block-aligned indices (bid_m, bid_n are integers)
    a = ct.load(a_tensor, index=(bid_m, k), shape=(TILE_M, TILE_K))  # ✓ TMA enabled by default
```

**TMA FAILS** (Non-Aligned Ragged Access):
```python
# Example: Ragged BMM with segment start at arbitrary offset
# Segment starts: [0, 5504, 10656, 14424] ← 10656 % 128 = 32 (NOT aligned!)

@ct.kernel
def ragged_kernel(...):
    # m_start = 10656 (not aligned to TILE_M=128)
    # Need rows [10656, 10784) but TMA can only load:
    #   - Block 83: rows [10624, 10752) ✗ Wrong offset
    #   - Block 84: rows [10752, 10880) ✗ Misses data

    # ✗ CANNOT express this with TMA block indices!
```

**SOLUTION: Use Gather/Scatter**:
```python
@ct.kernel
def ragged_kernel(...):
    # Calculate exact element indices
    m_indices = m_start + bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    # m_indices = [10656, 10657, ..., 10783] ← Exact rows needed!

    # Gather supports arbitrary element offsets
    a_tile = ct.gather(a, (m_indices_2d, k_indices_2d))  # ✓ Works! (padding defaults to 0)
```

**DECISION TREE**:
```
Is data access pattern block-aligned?
├─ YES → Use TMA (faster, hardware-accelerated)
│         Example: Regular GEMM, batch operations
│
└─ NO → Use gather/scatter (flexible, handles any offset)
          Examples: Ragged BMM, paged attention, sparse ops

Special case: Mixed approach
- Use TMA for aligned dimensions (e.g., B matrix in ragged BMM)
- Use gather/scatter for ragged dimensions (e.g., A, C matrices)
```

---

## Ragged BMM Lesson

```python
# Input: Single flattened tensor with segment boundaries
# a: Tensor[16384, 4096] with segments at [0, 5504, 10656, 14424, 16384]
#                                          ↑    ↑      ↑ unaligned offsets

# ✓ CORRECT: Hybrid approach
@ct.kernel
def ragged_bmm_kernel(...):
    # A, C are ragged → use gather/scatter
    m_indices = m_start + bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    a_tile = ct.gather(a, (m_indices_2d, k_indices_2d))  # padding defaults to 0

    # B is NOT ragged → use TMA (enabled by default)
    b_tile = ct.load(b, index=(bid_q, k, bid_n), shape=(1, TILE_K, TILE_N))

    acc = ct.mma(a_tile, b_tile, acc=acc)

    # C is ragged → use scatter
    ct.scatter(c, (m_indices_2d, n_indices_2d), c_tile)
```

---

## Group GEMM Works Differently

```python
# Input: List[Tensor] - each tensor has its own base pointer
# group_A = [A0, A1, A2, A3] ← Each starts at offset 0 within its own Array

# TMA works because each tensor is independently aligned
for g in range(group_size):
    a_tile = ct.load(As[g], index=(bid_m, k), ...)  # ✓ Always from offset 0!
```

**KEY INSIGHT**:
- **Group GEMM**: Multiple base pointers → TMA works
- **Ragged BMM**: Single base pointer + arbitrary offsets → Need gather/scatter

---

## Visual Difference

```
Group GEMM (Multiple Bases):
  A0 @ 0x1000 ──┐
  A1 @ 0x2000 ──┼─► TMA index=(0,0) for each ✓
  A2 @ 0x3000 ──┘

Ragged BMM (Single Base):
  ┌────────────────┐ @ 0x1000
  │ Segment 0      │ ← offset 0     (0%128=0) ✓
  ├────────────────┤ ← offset 5504  (5504%128=0) ✓
  │ Segment 1      │
  ├────────────────┤ ← offset 10656 (10656%128=32) ✗ NOT aligned!
  │ Segment 2      │
  └────────────────┘

  TMA can only index multiples of 128 from 0x1000!
```
