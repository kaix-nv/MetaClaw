# Section 7: Real-World Case Studies

> Part of the cuTile Performance Optimization Guide. Originally authored by **Yifei Song** and **Zhengyi Zhang**.

---

## Case Study 1: Softmax Optimization

**Before**:
```python
@ct.kernel
def softmax_kernel(...):
    bid_row = ct.bid(0)  # One block per row
    row = ct.load(input, index=(bid_row, 0), ...)
    ...

grid = (n_rows, 1, 1)  # Could be 100,000 blocks!
```
**Result**: 801-2029 GB/s (50-75% slower than Triton)

**After**:
```python
@ct.kernel(occupancy=4)
def softmax_kernel(..., n_rows: ct.Constant[int], ...):
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    for row_idx in range(pid, n_rows, num_programs):  # Persistent loop
        row = ct.load(input, index=(row_idx, 0), ...)
        ...

NUM_SM = torch.cuda.get_device_properties(device).multi_processor_count
grid = (NUM_SM * 4, 1, 1)  # Fixed ~600 blocks
```
**Result**: 1986-4367 GB/s (50-300% improvement, now competitive with Triton!)

**Key Lessons**:
- Static persistent gave 2-4x speedup
- Small workloads (N=1024-2048) saw biggest gains (256-300%)
- Large workloads (N=16384) saw smaller gains (saturate memory bandwidth)

---

## Case Study 2: Ragged BMM - Why TMA Failed

**Problem**: Flattened tensor with non-aligned segment boundaries.

**Data Structure**:
```
a: Tensor[16384, 4096]  (single contiguous array)
Segments: [0, 5504, 10656, 14424, 16384]
                    ↑ 10656 % 128 = 32 (NOT block-aligned!)
```

**Attempted Solution 1: TMA with Block Offset**
```python
m_block_offset = m_start // TILE_M  # 10656 // 128 = 83
a_tile = ct.load(a, index=(m_block_offset + bid_m, k),
                 shape=(TILE_M, TILE_K))
# ✗ Loads rows [83*128, 84*128) = [10624, 10752)
# But we need [10656, 10784) → 32-row misalignment!
```
**Result**: 97.71% accuracy (2.3% data corruption)

**Working Solution: Gather/Scatter**
```python
# Calculate exact element indices
m_start_int32 = ct.astype(m_start_scalar, ct.int32)
m_base = ct.broadcast_to(ct.reshape(m_start_int32, (1,)), (TILE_M,))
m_offset = ct.full((TILE_M,), bid_m * TILE_M, dtype=ct.int32) + ct.arange(TILE_M, dtype=ct.int32)
m_indices = m_base + m_offset  # [10656, 10657, ..., 10783] ← Exact!

# Broadcast and gather (padding_value defaults to 0)
m_2d = ct.reshape(m_indices, (TILE_M, 1))
m_bcast = ct.broadcast_to(m_2d, (TILE_M, TILE_K))
a_tile = ct.gather(a, (m_bcast, k_bcast))  # ✓ 100% correct
```
**Result**: 100% accuracy, all tests pass

**Key Lessons**:
- TMA fundamentally cannot handle non-aligned offsets
- Gather/scatter adds ~5-10% overhead but ensures correctness
- Hybrid approach: TMA for aligned, gather for ragged

---

## Case Study 3: Group GEMM vs Ragged BMM

**Why Group GEMM Can Use TMA**:
```python
# Input: List of independent tensors
group_A = [A0 @ 0x1000,  # ← Each has own base pointer
           A1 @ 0x2000,
           A2 @ 0x3000]

# TMA index always relative to current tensor's base
for g in range(group_size):
    a = ct.load(As[g], index=(bid_m, k), ...)
    # ✓ index=(0,0) means "start of As[g]", which is always aligned!
```

**Why Ragged BMM Cannot Use TMA**:
```python
# Input: Single flattened tensor with embedded segments
a: Tensor[16384, 4096] @ 0x1000  # ← ONE base pointer for all
Segments embedded at: [0, 5504, 10656, ...]
                              ↑ Must index into middle of array

# TMA index is absolute from single base
a = ct.load(a, index=(83, k), ...)
# ✗ 83*128=10624, but we need offset 10656!
```

**Visual Difference**:
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
