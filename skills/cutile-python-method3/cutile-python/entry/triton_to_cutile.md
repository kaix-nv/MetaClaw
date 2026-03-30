# Triton to cuTile Translation Reference

This reference provides mappings and patterns for translating Triton kernels to cuTile.

## Critical cuTile Requirements

When translating from Triton, remember these cuTile rules:
1. **Tile indices, not element indices**: `ct.load(A, index=(bid,), shape=(BLOCK,))` — the index is the tile index, not `bid * BLOCK`
2. **All tile dimensions must be powers of 2**: Use `2**((size-1).bit_length())` to round up
3. **All constants need type annotations**: `BLOCK: ct.Constant[int]` is required for compilation
4. **Use `ct.cdiv` for grid calculations**: `grid = (ct.cdiv(N, BLOCK),)`

## Triton to cuTile Mappings

### Core Constructs
| Triton | cuTile |
|--------|--------|
| `@triton.jit` | `@ct.kernel` |
| `tl.program_id(0)` | `ct.bid(0)` |
| `tl.program_id(1)` | `ct.bid(1)` |
| `tl.num_programs(0)` | Pass as `ct.Constant[int]` parameter |
| `tl.constexpr` | `ct.Constant[type]` |

### Memory Operations
| Triton | cuTile |
|--------|--------|
| `tl.load(ptr + offsets, mask=mask)` | `ct.load(tensor, index=(tile_idx,), shape=(BLOCK,))` |
| `tl.store(ptr + offsets, value, mask=mask)` | `ct.store(tensor, index=(tile_idx,), tile=value)` |
| `tl.arange(0, BLOCK)` | `ct.arange((BLOCK,), ct.int32)` |
| `tl.zeros((M, N), dtype)` | `ct.zeros((M, N), dtype)` |

### Math Operations
| Triton | cuTile |
|--------|--------|
| `tl.exp(x)` | `ct.exp(x)` |
| `tl.log(x)` | `ct.log(x)` |
| `tl.sqrt(x)` | `ct.sqrt(x)` |
| `tl.sin(x)` | `ct.sin(x)` |
| `tl.cos(x)` | `ct.cos(x)` |
| `tl.abs(x)` | `ct.abs(x)` |
| `tl.maximum(x, y)` | `ct.maximum(x, y)` |
| `tl.minimum(x, y)` | `ct.minimum(x, y)` |

### Reductions
| Triton | cuTile |
|--------|--------|
| `tl.sum(x, axis=0)` | `ct.sum(x, axis=0)` |
| `tl.max(x, axis=0)` | `ct.max(x, axis=0)` |
| `tl.min(x, axis=0)` | `ct.min(x, axis=0)` |

### Matrix Operations
| Triton | cuTile |
|--------|--------|
| `tl.dot(a, b)` | `ct.mma(a, b, acc)` |
| `tl.trans(x)` | `ct.transpose(x)` |

### Data Types
| Triton | cuTile |
|--------|--------|
| `tl.float32` | `ct.float32` |
| `tl.float16` | `ct.float16` |
| `tl.bfloat16` | `ct.bfloat16` |
| `tl.int32` | `ct.int32` |
| `tl.int64` | `ct.int64` |
| `tl.int1` / bool | `ct.bool_` |

### Activation Functions
| Triton | cuTile |
|--------|--------|
| `tl.sigmoid(x)` | `1.0 / (1.0 + ct.exp(-x))` |
| `x * tl.sigmoid(x)` (SiLU) | `x * (1.0 / (1.0 + ct.exp(-x)))` |

### Control Flow
| Triton | cuTile |
|--------|--------|
| `tl.where(cond, x, y)` | `ct.where(cond, x, y)` |
| `mask = offsets < N` | Handle via grid sizing or `ct.where` |

## Key Translation Differences

### 1. Memory Access Model

**Triton** uses pointer arithmetic with explicit masks:
```python
offsets = pid * BLOCK + tl.arange(0, BLOCK)
mask = offsets < N
x = tl.load(ptr + offsets, mask=mask)
```

**cuTile** uses tile-based indexing:
```python
bid = ct.bid(0)
x = ct.load(tensor, index=(bid,), shape=(BLOCK,))
```

### 2. Grid Launch

**Triton**:
```python
grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
kernel[grid](ptr, N, BLOCK=256)
```

**cuTile**:
```python
grid = (ct.cdiv(N, BLOCK),)
ct.launch(torch.cuda.current_stream(), grid, kernel, (tensor, output, N, BLOCK))
```

### 3. Autotuning

**Triton**: Uses `@triton.autotune` decorator with configs
**cuTile**: Pass configurations as `ct.Constant` parameters; autotuning handled externally

### 4. Mask Handling

**Triton**: Explicit mask parameters in load/store
**cuTile**: Rely on grid sizing to avoid out-of-bounds; use `ct.where` for conditional operations

## Translation Example

**Triton**:
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

**cuTile**:
```python
@ct.kernel
def add_kernel(x, y, out, N: ct.Constant[int], BLOCK: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(BLOCK,))
    ty = ct.load(y, index=(bid,), shape=(BLOCK,))
    ct.store(out, index=(bid,), tile=tx + ty)
```

## Common Pitfalls

1. **Pointer arithmetic**: cuTile doesn't use pointers; convert to index-based access
2. **Masks**: cuTile handles boundaries via grid sizing, not explicit masks
3. **`tl.dot` vs `ct.mma`**: cuTile's `ct.mma` requires an accumulator argument
4. **Autotuning configs**: Extract optimal config or pass as `ct.Constant` parameters
5. **Element indices**: Triton uses `pid * BLOCK`, cuTile uses tile index `(bid,)` directly
