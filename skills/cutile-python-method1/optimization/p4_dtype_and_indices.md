# Section 4-5: DType Consistency and Gather/Scatter Index Calculation

> Part of the cuTile Performance Optimization Guide. Originally authored by **Yifei Song** and **Zhengyi Zhang**.

---

## DType Consistency

**Use ct dtypes for type specifications (e.g., `ct.int32`, `ct.float32`):**

```python
# ✅ CORRECT - Use ct.xxx dtypes
indices = ct.full((TILE_M,), offset, dtype=ct.int32)
acc = ct.full((tm, tn), 0., dtype=ct.float32)
mask = ct.full((tm,), True, dtype=ct.bool_)

# For arange, use ct.int32:
offs = ct.arange(TILE_M, dtype=ct.int32)
```

**NOTE**: Use `tensor.dtype` when converting to match an existing tensor's type:
```python
# Use output tensor's dtype for final conversion
result = ct.astype(accumulator, C.dtype)
```

---

## Gather/Scatter Index Calculation

**CRITICAL**: Use Python operators (`+`, `-`, `*`) for index calculation, NOT `ct.add/ct.mul`.

**WRONG** (ct operations promote to float):
```python
offset = ct.mul(pid_m, stride)  # Returns Tile[float32, ...]
indices = ct.add(base, offset)  # Tile[float32, ...] ✗ Invalid for gather!
data = ct.gather(array, indices, ...)  # ✗ TileTypeError!
```

**RIGHT** (Python operators preserve int32):
```python
offset = pid_m * stride  # Python int or Tile[int32, ...]
indices = base + offset  # Tile[int32, ...] ✓
data = ct.gather(array, indices, ...)  # ✓ Works!
```

---

## Complete Example

```python
# Calculate 2D indices for gather
m_base = ct.full((TILE_M,), m_start, dtype=ct.int32)
m_offset = ct.full((TILE_M,), bid_m * TILE_M, dtype=ct.int32) + ct.arange(TILE_M, dtype=ct.int32)
m_indices = m_base + m_offset  # ✓ Python + preserves int32

k_indices = ct.full((TILE_K,), k * TILE_K, dtype=ct.int32) + ct.arange(TILE_K, dtype=ct.int32)

# Broadcast to 2D
m_2d = ct.reshape(m_indices, (TILE_M, 1))
k_2d = ct.reshape(k_indices, (1, TILE_K))
m_bcast = ct.broadcast_to(m_2d, (TILE_M, TILE_K))
k_bcast = ct.broadcast_to(k_2d, (TILE_M, TILE_K))

# Gather (padding_value defaults to 0)
a_tile = ct.gather(a, (m_bcast, k_bcast))  # ✓
```
