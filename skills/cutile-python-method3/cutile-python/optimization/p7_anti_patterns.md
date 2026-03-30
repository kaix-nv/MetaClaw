# Section 8: Performance Anti-Patterns

> Part of the cuTile Performance Optimization Guide. Originally authored by **Yifei Song** and **Zhengyi Zhang**.

---

## Anti-Pattern 1: Excessive Type Conversions

```python
# ✗ BAD: Convert for every row in loop
for row in range(...):
    row_fp32 = ct.astype(row, ct.float32)
    result = compute(row_fp32)
    row_fp16 = ct.astype(result, ct.float16)

# Better: Keep in fp32 longer, batch conversions
```

---

## Anti-Pattern 2: Redundant Tensor Creation

```python
# ✗ BAD: Create mask inside loop
for i in range(n):
    mask = ct.full((tm,), True, dtype=ct.bool_)  # Recreated every iteration!

# ✓ GOOD: Create once outside loop
mask = ct.full((tm,), True, dtype=ct.bool_)
for i in range(n):
    # Use mask
```

---

## Anti-Pattern 3: Column Loops for Row-Wise Ops

```python
# ✗ BAD: Softmax with column loop
for col_tile in range(num_col_tiles):
    partial = ct.load(..., index=(row, col_tile), ...)
    # Partial softmax on tile → WRONG! Need full row for correct normalization

# ✓ GOOD: Load entire row
row = ct.load(..., index=(row, 0), shape=(1, TILE_SIZE_COVERS_ALL_COLS))
# Compute softmax on complete row
```

---

## Anti-Pattern 4: Host-Side Padding Overhead

```python
# ✗ BAD: Pad before every kernel call
for iteration in training_loop:
    input_padded = torch.nn.functional.pad(input, ...)  # Memory allocation!
    ct.launch(kernel, (input_padded, ...))

# ✓ GOOD: Pad once, reuse
input_padded = torch.nn.functional.pad(input, ...) if needs_pad else input
for iteration in training_loop:
    ct.launch(kernel, (input_padded, ...))
```
