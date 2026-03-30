<!-- Auto-generated from cutile-python v1.2.0 by maintain.py. Do not edit manually. -->

# Performance Tuning

Several performance tuning techniques are available in cuTile:

- architecture-specific configuration values, using ByTarget;
- load/store hints such as `latency` and `allow_tma`.

## Architecture-specific configuration

*class* cuda.tile.ByTarget(*\**, *default=UNSPECIFIED*, *\*\*value\_by\_target*)
:   Type used to specify a value that depends on the target GPU architecture.

    Parameters:
    :   - **default** – The fallback value to use when the target GPU architecture is not explicitly
          listed in `value_by_target`.
        - **value\_by\_target** – Mapping from GPU architecture name to value. Keys must be strings of
          the form `"sm_<major><minor>"`, such as `"sm_100"` or `"sm_120"`.

    Examples

    Use one `num_ctas` value for all architectures:

    ```
    from cuda.tile import kernel, ByTarget

    @kernel(num_ctas=8)
    def kernel_fn(x):
        ...
    ```

    Use different `num_ctas` values for specific architectures, and a
    fallback value for all others:

    ```
    from cuda.tile import kernel, ByTarget

    @kernel(num_ctas=ByTarget(sm_100=8, sm_120=4, default=2))
    def kernel_fn(x):
        ...
    ```

See Tile Kernels for the full description of kernel configuration
parameters such as `num_ctas`, `occupancy` and `opt_level`. Any of
these options may be given as a ByTarget value to specialize them
for different GPU architectures.

## Load/store performance hints

The load() and store() operations accept optional keyword
arguments that can influence how memory traffic is scheduled and lowered:

- `latency` (`int` or `None`) – A hint indicating how heavy the DRAM
  traffic will be for this operation. It shall be an integer between
  1 (low) and 10 (high). If it is `None` or not provided, the compiler
  will infer the latency.
- `allow_tma` (`bool` or `None`) – If `True`, the load or store may be
  lowered to use TMA (Tensor Memory Accelerator) when the target architecture
  supports it. If `False`, TMA will not be used for this operation.
  By default, TMA is allowed.

These hints are optional: kernels will compile and run without specifying
them, but providing them can help the compiler make better code-generation
decisions for a particular memory-access pattern.

### Example

```
import cuda.tile as ct

TILE_SIZE = 16

@ct.kernel
def load_store_with_hints_kernel(x, y):
    bid = ct.bid(0)
    tx = ct.load(
        x,
        index=(bid,),
        shape=(TILE_SIZE,),
        latency=8,        # high-latency DRAM load
    )
    ct.store(
        y,
        index=(bid,),
        tile=tx,
        latency=2,        # cheaper write
        allow_tma=False,  # disallow TMA
    )
```
