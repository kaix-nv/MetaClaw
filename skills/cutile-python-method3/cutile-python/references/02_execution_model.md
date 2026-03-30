<!-- Auto-generated from cutile-python v1.2.0 by maintain.py. Do not edit manually. -->

# Execution Model

## Abstract Machine

A *tile kernel* is executed by logical thread blocks that are organized in
a 1D, 2D, or 3D *grid*.

Each *block* is executed by a subset of a GPU, which is decided by the
implementation, not the programmer.
Each block executes the body of the kernel.
Scalar operations are executed serially by a single thread of the block,
and array operations are collectively executed in parallel by all threads of
the block.

Tile programs explicitly describe block-level parallelism, but not
thread-level parallelism.
Threads cannot be explicitly identified or manipulated in tile programs.

Explicit synchronization or communication within a block is not
permitted, but it is allowed between different blocks.

It is important to not confuse blocks (units of execution) with
tiles (units of data).
A block may work with multiple different tiles with
differing shapes originating from differing global arrays.

## Execution Spaces

cuTile code is executed on one or more *targets*, which are distinct execution environments that
are distinguished by different hardware resources or programming models.

A function is *usable* if it can be called.
A type or object is *usable* if its attributes are accessible (can be read and written) and its
methods are callable.

Some functions, types, and objects are only usable on certain *targets*.
The set of *targets* that such a construct is usable on is called its *execution space*.

- *Host code* is the execution space that includes all CPU targets.
- *SIMT code* is the execution space that includes all CUDA SIMT targets.
  Note: This has historically been called device code, but we avoid this term to prevent ambiguity.
- *Tile code* is the execution space that includes all CUDA tile targets.

Functions can have decorators that explicitly specify their execution space.
These are called *annotated functions*.

## Tile Functions

*class* cuda.tile.function(*func=None*, */*, *\**, *host=False*, *tile=True*)
:   *Tile functions* are functions that are usable in tile code.

    This decorator indicates what execution spaces a function can be called from.
    With no arguments, it denotes a tile-only function.

    When an unannotated function is called by a tile function, tile shall be added to the
    unannotated function’s execution space.
    This process is recursive.
    No explicit annotation is required.

    The types usable as parameters to a tile function are described in the data model.

    Parameters:
    :   - **host** (*bool**,* *optional*) – Whether the function can be called from host code.
          Default is False.
        - **tile** (*bool**,* *optional*) – Whether the function can be called from tile code.
          Default is True.

## Tile Kernels

*class* cuda.tile.kernel(*function=None*, */*, *\*\*kwargs*)
:   A *tile kernel* is a function executed by each block in a grid.

    Functions with this decorator are kernels.

    Kernels are the entry points of tile code.
    Their execution space shall be only tile code; they cannot be called from host code.

    Kernels cannot be called directly. Instead, use launch() to
    queue a kernel for execution over a grid.

    The types usable as parameters to a kernel are described in the data model.

    Parameters:
    :   - **num\_ctas** – Number of CTAs in a CGA. Must be a power of 2 between 1 and 16, inclusive.
          Default: None (auto).
        - **occupancy** – Expected number of active CTAs per SM, [1, 32]. Default: None (auto).
        - **opt\_level** – Optimization level [0, 3], default 3.

    Target-specific values for the compiler options above can be provided
    using a ByTarget object.

    Examples:

    ```
    @ct.kernel
    def f(a, b, c):
        pass

    grid = (8, 8)
    ct.launch(stream, grid, f, (A, B, C))
    ```

cuda.tile.launch(*stream*, *grid*, *kernel*, *kernel\_args*, */*)
:   Launch a cuTile kernel.

    Parameters:
    :   - **stream** – The CUDA stream to execute the kernel on.
        - **grid** – Tuple of up to 3 grid dimensions to execute the kernel over.
        - **kernel** – The kernel to execute.
        - **kernel\_args** – Positional arguments to pass to the kernel.

## Python Subset

Tile code supports a subset of the Python language.
Within tile code, there is no Python runtime.

Only Python features explicitly enumerated in this document are supported.
Many features, such as lambdas, exceptions, and coroutines are not supported today.

### Object Model & Lifetimes

All objects created within tile code are immutable.
Any operation that conceptually modifies an object or its attributes creates and returns a new
object.
Attributes cannot be dynamically added to objects.

The only mutable objects that can be used in tile code are arrays, which must be passed in as
kernel parameters.

The caller of a kernel must ensure that:

- No arrays passed to the kernel alias one another.
- All passed arrays remain valid until the kernel has finished execution.

### Control Flow

Python control flow statements (`if`, `for`, `while`, etc.) shall be usable in tile code.
They can be arbitrarily nested.

#### Current limitations

The Python subset used in tile code imposes additional restrictions on control flow:

- `step` must be strictly positive.

  Negative-step ranges such as
  `range(10, 0, -1)` are not supported today. Passing a negative step
  indirectly via a variable may lead to undefined behavior.

## Tile Parallelism

When a block executes a function that takes tiles as parameters, it may parallelize the
evaluation of the function across the block’s execution resources.
Unless otherwise specified, the execution shall complete before the function returns.

## Constantness

### Constant Expressions & Objects

Some facilities require certain parameters to be an object that is known statically at compilation time.
*Constant expressions* produce *constant objects* suitable for such parameters. Constant expressions are:

- A literal object.
- Integer arithmetic expressions where all the operands are literal objects.
- A local object or parameter that is assigned from a literal object or constant expression.
- A global object that is defined at the time of compilation or launch.

By default, numeric constants are *loosely typed*: until used in a context that requires
a type of a specific width, integer constants have infinite precision, and floating-point
constants are stored in the IEEE 754 double precision format.

A *strictly typed* constant can be created by calling a dtype object as a constructor,
e.g. `ct.int16(5)` creates a strictly typed `int16` constant. When a strictly typed constant
is combined with a loosely typed constant, the result is a strictly typed constant.
For example `ct.int16(5) + 2` will create a strictly typed `int16` constant 7.

Combining two strictly typed constants creates a new strictly typed constant. In this case,
the regular type promotion rules apply.
For example, `ct.int16(5) + ct.int32(7)` will create a strictly typed `int32` constant 12.

### Constant Embedding

If a parameter to a kernel is *constant embedded*, then:

- All uses of the parameter shall act as if they were replaced by the literal value of the parameter.
- There shall be a distinct machine representation of the kernel for each different value of the parameter that the kernel is invoked with. Note: The kernel shall be compiled once for each different value of the parameter, even if JIT caching is enabled.
- The machine representation of the parameter shall be 0 bytes.

### Constant Type Hints

```
import cuda.tile as ct
```

```
def needs_constant(x: ct.Constant):
    pass

def needs_constant_int(x: ct.Constant[int]):
    pass
```

*class* cuda.tile.ConstantAnnotation
:   A `typing.Annotated` metadata class indicating that an object shall be constant embedded.

    If an object of this class is passed as a metadata argument to a `typing.Annotated` type hint
    on a parameter, then the parameter shall be a constant embedded.

cuda.tile.Constant
:   A type hint indicating that a value shall be constant embedded.
    It can be used either with (`Constant[int]`) or without (`Constant`, meaning a constant of any
    type) an underlying type hint.

    alias of `Annotated`[`T`, ConstantAnnotation()]
