# Section 2: Occupancy and Autotune

> Part of the cuTile Performance Optimization Guide. Originally authored by **Yifei Song** and **Zhengyi Zhang**.

---

**PRINCIPLE**: Use autotuning to find optimal occupancy and tile configurations. Manual tuning is error-prone and workload-dependent.

---

## Autotune Parameter Ranges

**CRITICAL**: Design your search space to cover the full parameter ranges. The autotuner will find the best combination.

| Parameter | Valid Range | Description |
|-----------|-------------|-------------|
| **occupancy** | 1 - 32 | Number of active warps per SM. Higher = more parallelism, lower = more resources per warp |
| **num_ctas** | 1, 2, 4, 8, 16 | Number of CTAs (thread blocks) to fuse. Powers of 2 only |
| **TILE_SIZE** | Powers of 2 | Tile dimension size. Should cover from smallest expected input to largest |

---

## Autotune Template

### Step 1: Define Config Generator

```python
from types import SimpleNamespace
import torch

def _my_kernel_autotune_configs():
    """
    Autotune config generator.

    IMPORTANT: Cover a WIDE RANGE of configurations!
    - The autotuner will find the best combination
    - Don't pre-optimize by narrowing the search space
    """
    gpu_capability = torch.cuda.get_device_capability()

    # === Define your search ranges ===
    # Tile sizes: Cover from smallest expected input to largest
    # Adjust based on your kernel's expected input dimensions
    tile_sizes = [...]  # e.g., powers of 2 covering your input range

    # Occupancy: Range is [1, 32]
    # Include a spread of values - autotuner will find optimal
    occupancies = [...]  # Select a subset from 1-32

    # num_ctas: Valid values are 1, 2, 4, 8, 16
    num_ctas_options = [...]  # Select from [1, 2, 4, 8, 16]

    # Generate all combinations
    for tile in tile_sizes:
        for occ in occupancies:
            for num_ctas in num_ctas_options:
                yield SimpleNamespace(
                    TILE_SIZE=tile,
                    num_ctas=num_ctas,
                    occupancy=occ,
                )
```

### Step 2: Autotune Launch Function

```python
import cuda.tile_experimental as ct_experimental

def _my_kernel_autotune_base(stream, input, output, N, C):
    """
    Autotuned kernel launch with dynamic grid and args.
    """
    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count

    def args_fn(cfg):
        # Clamp TILE_SIZE to actual dimension if needed
        tile_size = min(cfg.TILE_SIZE, _next_power_of_2(C))
        return (input, output, tile_size, N)

    def grid_fn(cfg):
        # Static persistent scheduling
        num_programs = min(NUM_SM * cfg.occupancy, N)
        return (num_programs, 1, 1)

    ct_experimental.autotune_launch(
        stream,
        grid_fn=grid_fn,
        kernel=_my_kernel,
        args_fn=args_fn,
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=_my_kernel_autotune_configs,
    )
```

### Step 3: Default Configs (Non-Autotune Fallback)

```python
def _get_default_kernel_configs():
    """GPU-specific defaults when autotune is disabled."""
    # Use middle-ground values that work reasonably for most cases
    return {"TILE_SIZE": 256, "num_ctas": 1, "occupancy": 4}
```

### Step 4: Conditional Autotune in Forward Pass

```python
import os

class MyOpCuTile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ...):
        # Autotune enabled by default
        enable_autotune = os.environ.get("DISABLE_CUTILE_TUNE", "0") != "1"

        if enable_autotune:
            _my_kernel_autotune_base(
                torch.cuda.current_stream(), x, output, N, C
            )
        else:
            # Use fixed default configs
            configs = _get_default_kernel_configs()
            kernel = build_cutile_kernel_from_autotune(
                _my_kernel._pyfunc,
                kernel_configs=configs,
                device=x.device,
            )
            ct.launch(stream, grid, kernel, args)

        return output
```

---

## Guidelines for Choosing Search Ranges

### Tile Sizes
- **Minimum**: Should be ≤ smallest expected input dimension (avoid wasted computation)
- **Maximum**: Should be ≥ largest expected input dimension (for efficiency)
- **Spacing**: Use powers of 2, include enough intermediate values

### Occupancy (1-32)
- **Wider range**: Better for kernels that will see diverse workloads
- **Narrower range**: OK if you know your kernel's characteristics well
- **Rule of thumb**: Include at least low (1-4), medium (8-16), and high (20-32) values

### num_ctas (1, 2, 4, 8, 16)
- **Start with 1**: Most kernels work well with single CTA
- **Add more**: If your kernel benefits from CTA cooperation
- **Full range**: When uncertain, include all options

---

## Search Space Size Tradeoffs

| Search Space | First-Run Time | When to Use |
|--------------|----------------|-------------|
| Large (50+ configs) | Minutes | Production, unknown workloads |
| Medium (15-30 configs) | ~30-60 sec | Development, known workload range |
| Small (5-10 configs) | ~10-20 sec | Quick iteration, narrow use case |
| Fixed (1 config) | 0 | Debugging with `DISABLE_CUTILE_TUNE=1` |

**TIP**: Start with a larger search space, then narrow based on profiling results if compilation time is a concern.

---

## Manual Occupancy (When Not Using Autotune)

If autotune is disabled, match kernel decorator occupancy with launch grid:

```python
@ct.kernel(occupancy=4)
def my_kernel(...):
    ...

# Launch matches kernel
NUM_SM = torch.cuda.get_device_properties(device).multi_processor_count
occupancy = 4  # ← SAME as kernel decorator
grid = (NUM_SM * occupancy, 1, 1)
```

**OCCUPANCY HINTS** (for manual tuning):
- `occupancy=1-4`: Compute-bound kernels (heavy math)
- `occupancy=4-8`: Balanced kernels (GEMM, TMA operations)
- `occupancy=8-16`: Memory-bound kernels (reductions, element-wise)
- `occupancy=16-32`: Very light kernels (simple copies, casts)
