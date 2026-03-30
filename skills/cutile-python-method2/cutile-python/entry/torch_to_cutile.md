# PyTorch to cuTile Translation Reference

This reference provides mappings and patterns for translating PyTorch code to cuTile kernels.

## Critical cuTile Requirements

When translating from PyTorch, remember these cuTile rules:
1. **Tile indices, not element indices**: `ct.load(A, index=(bid,), shape=(BLOCK,))` — the index is the tile index, not `bid * BLOCK`
2. **All tile dimensions must be powers of 2**: Use `2**((size-1).bit_length())` to round up
3. **All constants need type annotations**: `BLOCK: ct.Constant[int]` is required for compilation
4. **Use `ct.cdiv` for grid calculations**: `grid = (ct.cdiv(N, BLOCK),)`

## PyTorch to cuTile Mappings

### Basic Operations
| PyTorch | cuTile |
|---------|--------|
| `x + y` | `tx + ty` (on loaded tiles) |
| `x * y` | `tx * ty` (on loaded tiles) |
| `x - y` | `tx - ty` (on loaded tiles) |
| `x / y` | `tx / ty` (on loaded tiles) |
| `torch.exp(x)` | `ct.exp(tx)` |
| `torch.log(x)` | `ct.log(tx)` |
| `torch.sqrt(x)` | `ct.sqrt(tx)` |
| `torch.sin(x)` | `ct.sin(tx)` |
| `torch.cos(x)` | `ct.cos(tx)` |
| `torch.tanh(x)` | `ct.tanh(tx)` |
| `torch.abs(x)` | `ct.abs(tx)` |

### Activation Functions
| PyTorch | cuTile |
|---------|--------|
| `torch.relu(x)` | `ct.maximum(tx, 0.0)` |
| `torch.sigmoid(x)` | `1.0 / (1.0 + ct.exp(-tx))` |
| `x * torch.sigmoid(x)` (SiLU) | `tx * (1.0 / (1.0 + ct.exp(-tx)))` |

### Reductions
| PyTorch | cuTile |
|---------|--------|
| `torch.sum(x, dim=-1)` | `ct.sum(tx, axis=-1)` |
| `torch.max(x, dim=-1)` | `ct.max(tx, axis=-1)` |
| `torch.min(x, dim=-1)` | `ct.min(tx, axis=-1)` |

### Scan Operations
| PyTorch | cuTile |
|---------|--------|
| `torch.cumsum(x, dim=-1)` | `ct.cumsum(tx, axis=-1)` |
| `torch.cumprod(x, dim=-1)` | `ct.cumprod(tx, axis=-1)` |

### Matrix Operations
| PyTorch | cuTile |
|---------|--------|
| `A @ B` | `ct.mma(a_tile, b_tile, acc)` |
| `torch.matmul(A, B)` | `ct.mma(a_tile, b_tile, acc)` |
| `A.T` / `A.transpose(-1, -2)` | `ct.transpose(a_tile)` |

### Data Types
| PyTorch | cuTile |
|---------|--------|
| `torch.float32` | `ct.float32` |
| `torch.float16` | `ct.float16` |
| `torch.bfloat16` | `ct.bfloat16` |
| `torch.int32` | `ct.int32` |
| `torch.int64` | `ct.int64` |
| `torch.bool` | `ct.bool_` |

**Default data types** (when the user does not specify): use `float16` for input tensors and `float32` for accumulators. For example, in a matmul kernel, load inputs as `float16` and accumulate into a `float32` tile before storing the result.

## Key Translation Patterns

### Pattern 1: Element-wise Operations
**PyTorch**:
```python
def forward(x, y):
    return torch.exp(x) + y
```

**cuTile**:
```python
@ct.kernel
def elementwise_kernel(x, y, output, N: ct.Constant[int], BLOCK: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(BLOCK,))
    ty = ct.load(y, index=(bid,), shape=(BLOCK,))
    result = ct.exp(tx) + ty
    ct.store(output, index=(bid,), tile=result)
```

### Pattern 2: Reductions
**PyTorch**:
```python
def forward(x):
    return torch.sum(x, dim=-1)
```

**cuTile**: Load a tile, reduce within the tile using `ct.sum(tile, axis=-1)`.

### Pattern 3: Matrix Multiplication
**PyTorch**:
```python
def forward(A, B):
    return A @ B
```

**cuTile**: Use `ct.mma(a_tile, b_tile, accumulator)` with proper tiling over M, N, K dimensions.

## Tensor Preparation

Before passing PyTorch tensors to cuTile kernels:
- Ensure tensors are on CUDA: `.cuda()` or `.to("cuda")`
- Remove gradients if present: use `.data` attribute
- Ensure contiguity: `.contiguous()` if needed
- Verify compatible dtypes (float32, float16, bfloat16, int32, etc.)

## Common Pitfalls

1. **Forgetting `.data`**: cuTile doesn't support tensors with gradients
2. **Non-contiguous tensors**: Call `.contiguous()` before passing to kernel
3. **Wrong dtype**: Ensure PyTorch dtype matches cuTile dtype
4. **Implicit broadcasting**: PyTorch broadcasts automatically; cuTile requires explicit handling
5. **In-place operations**: PyTorch `x.add_(y)` needs separate input/output in cuTile
