<!-- Auto-generated from cutile-python v1.2.0 by maintain.py. Do not edit manually. -->

# Data Model

cuTile is an array-based programming model.
The fundamental data structure is multidimensional arrays with elements of a single homogeneous type.
cuTile Python does not expose pointers, only arrays.

An array-based model was chosen because:

- Arrays know their bounds, so accesses can be checked to ensure safety and correctness.
- Array-based load/store operations can be efficiently lowered to speed-of-light hardware mechanisms.
- Python programmers are used to array-based programming frameworks such as NumPy.
- Pointers are not a natural choice for Python.

Within tile code, only the types described in this section are supported.

## Global Arrays

A *global array* (or *array*) is a container of elements of a specific dtype
arranged in a logical multidimensional space.

Array’s *shape* is a tuple of integer values, each denoting the length of
the corresponding dimension.
The length of the shape tuple equals the arrays’s number of dimensions.
The product of shape values equals the total logical number of elements in the array.

Arrays are stored in global memory using a *strided memory layout*: in addition to a shape,
an array also has an equally sized tuple of *strides*. Strides determine the mapping of logical
array indices to physical memory locations. For example, for a 3-dimensional float32 array
with strides (s1, s2, s3), the memory address of the element at the logical index
(i1, i2, i3) will be:

```
base_addr + 4 * (s1 * i1 + s2 * i2 + s3 * i3),
```

where `base_addr` is the base address of the array and 4 is the byte size of a single float32
element.

New arrays can only be allocated by the host, and passed to the tile kernel as arguments.
Tile code can only create new views of existing arrays, for example using
Array.slice(). Like in Python, assigning an array object to another variable does not copy
the underlying data, but creates another reference to the array object.

Any object that implements the DLPack interface or the CUDA Array Interface
can be passed to the kernel as an argument. Example: CuPy arrays and PyTorch tensors.

If two or more array arguments are passed to the kernel, their memory storage must not overlap.
Otherwise, behavior is undefined.

Array’s shape can be queried using the Array.shape attribute, which
returns a tuple of int32 scalars. These scalars are non-constant, runtime values.
Using int32 makes the tile code more performant at the cost of limiting the maximum
representable shape at 2,147,483,647 elements. This limitation will be lifted in the future.

See also

cuda.tile.Array class documentation

## Tiles and Scalars

A *tile* is an immutable multidimensional collection of elements of a specific dtype.

Tile’s *shape* is a tuple of integer values, each denoting the length of the corresponding dimension.
The length of the shape tuple equals the tile’s number of dimensions.
The product of shape values equals the total number of elements in the tile.

The shape of a tile must be known at compile time. Each dimension of a tile must be a power of 2.

Tile’s dtype and shape can be queried with the `dtype` and `shape` attributes, respectively.
For example, if `x` is a float32 tile, the expression `x.dtype` will return
a compile-time constant equal to cuda.tile.float32.

A zero-dimensional tile is called a *scalar*. Such tile has exactly one element. The shape
of a scalar is the empty tuple (). Numeric literals like 7 or 3.14 are treated as
constant scalars, i.e. zero-dimensional tiles.

Since scalars are tiles, they slightly differ in behavior from Python’s `int`/`float` objects.
For example, they have `dtype` and `shape` attributes:

```
a = 0
# The following line will evaluate to cuda.tile.int32 in cuTile,
# but would raise an AttributeError in Python:
a.dtype
```

Tiles can only be used in tile code, not host code.
The contents of a tile do not necessarily have a physical representation in memory.
Non-scalar tiles can be created by loading from global arrays using functions such as
cuda.tile.load() and cuda.tile.gather() or with factory functions
such as cuda.tile.zeros().

Tiles can also be stored into global arrays using functions such as cuda.tile.store()
or cuda.tile.scatter().

Only scalars (i.e. 0-dimensional tiles) can be used as kernel parameters.

Scalar constants are loosely typed by default, for example, a literal `2` or
a constant attribute like `Tile.ndim`, `Tile.shape`, or `Array.ndim`.

See also

cuda.tile.Tile class documentation

## Element & Tile Space

![_images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__dark_background.svg](_images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__dark_background.svg)
![_images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__light_background.svg](_images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__light_background.svg)
![_images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__dark_background.svg](_images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__dark_background.svg)
![_images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__light_background.svg](_images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__light_background.svg)

The *element space* of an array is the multidimensional space of elements contained in that array,
stored in memory according to a certain layout (row major, column major, etc).

The *tile space* of an array is the multidimensional space of tiles into that array of a certain
tile shape.
A tile index `(i, j, ...)` with shape `S` refers to the elements of the array that belong to the
`(i+1)`-th, `(j+1)`-th, … tile.

When accessing the elements of an array using tile indices, the multidimensional memory layout of the array is used.
To access the tile space with a different memory layout, use the order parameter of load/store operations.

## Shape Broadcasting

*Shape broadcasting* allows tiles with different shapes to be combined in arithmetic operations.
When performing operations between tiles of different shapes, the smaller tile is automatically
extended to match the shape of the larger one, following these rules:

- Tiles are aligned by their trailing dimensions.
- If the corresponding dimensions have the same size or one of them is 1, they are compatible.
- If one tile has fewer dimensions, its shape is padded with 1s on the left.

Broadcasting follows the same semantics as NumPy, which makes code more concise and readable
while maintaining computational efficiency.

## Data Types

*class* cuda.tile.DType
:   A *data type* (or *dtype*) describes the type of the objects of an array, tile, or
    operation.

    Dtypes determine how values are stored in memory and how operations on those values are
    performed.
    Dtypes are immutable.

    Dtypes can be used in host code and tile code.
    They can be kernel parameters.

    *property* bitwidth
    :   The number of bits in an element of the data type.

    *property* name
    :   The name of the data type.

cuda.tile.bool\_
:   A 8-bit arithmetic dtype (`True` or `False`).

cuda.tile.uint8
:   A 8-bit unsigned integer arithmetic dtype whose values exist on the interval [0, +256].

cuda.tile.uint16
:   A 16-bit unsigned integer arithmetic dtype whose values exist on the interval [0, +65,536].

cuda.tile.uint32
:   A 32-bit unsigned integer arithmetic dtype whose values exist on the interval [0, +4,294,967,295].

cuda.tile.uint64
:   A 64-bit unsigned integer arithmetic dtype whose values exist on the interval [0, +18,446,744,073,709,551,615].

cuda.tile.int8
:   A 8-bit signed integer arithmetic dtype whose values exist on the interval [−128, +127].

cuda.tile.int16
:   A 16-bit signed integer arithmetic dtype whose values exist on the interval [−32,768, +32,767].

cuda.tile.int32
:   A 32-bit signed integer arithmetic dtype whose values exist on the interval [−2,147,483,648, +2,147,483,647].

cuda.tile.int64
:   A 64-bit signed integer arithmetic dtype whose values exist on the interval [−9,223,372,036,854,775,808, +9,223,372,036,854,775,807].

cuda.tile.float16
:   A IEEE 754 half-precision (16-bit) binary floating-point arithmetic dtype (see IEEE 754-2019).

cuda.tile.float32
:   A IEEE 754 single-precision (32-bit) binary floating-point arithmetic dtype (see IEEE 754-2019).

cuda.tile.float64
:   A IEEE 754 double-precision (64-bit) binary floating-point arithmetic dtype (see IEEE 754-2019).

cuda.tile.bfloat16
:   A 16-bit floating-point arithmetic dtype with 1 sign bit, 8 exponent bits, and 7 mantissa bits.

cuda.tile.tfloat32
:   A 32-bit tensor floating-point numeric dtype with 1 sign bit, 8 exponent bits, and 10 mantissa bits (19-bit representation stored in 32-bit container).

cuda.tile.float8\_e4m3fn
:   An 8-bit floating-point numeric dtype with 1 sign bit, 4 exponent bits, and 3 mantissa bits.

cuda.tile.float8\_e5m2
:   An 8-bit floating-point numeric dtype with 1 sign bit, 5 exponent bits, and 2 mantissa bits.

## Numeric & Arithmetic Data Types

A *numeric* data type represents numbers. An *arithmetic* data type is a numeric data type
that supports general arithmetic operations such as addition, subtraction, multiplication,
and division.

## Arithmetic Promotion

Binary operations can be performed on two tile or scalar operands of different numeric dtypes.

When both operands are loosely typed numeric constants, then the result is also
a loosely typed constant: for example, `5 + 7` is a loosely typed integral constant 12,
and `5 + 3.0` is a loosely typed floating-point constant 8.0.

If any of the operands is not a loosely typed numeric constant, then both are *promoted*
to a common dtype using the following process:

- Each operand is classified into one of the three categories:
  *boolean*, *integral*, or *floating-point*.
  The categories are ordered as follows: *boolean* < *integral* < *floating-point*.
- If either operand is a loosely typed numeric constant, a concrete dtype is picked for it:
  integral constants are treated as int32, int64, or uint64, depending on the value;
  floating-point constants are treated as float32.
- If one of the two operands has a higher category than the other, then its concrete dtype
  is chosen as the common dtype.
- If both operands are of the same category, but one of them is a loosely typed numeric constant,
  then the other operand’s dtype is picked as the common dtype.
- Otherwise, the common dtype is computed according to the table below.

|  | b1 | u8 | u16 | u32 | u64 | i8 | i16 | i32 | i64 | f16 | f32 | f64 | bf | tf32 | f8e4m3fn | f8e5m2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1 | b1 | u8 | u16 | u32 | u64 | i8 | i16 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| u8 | u8 | u8 | u16 | u32 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| u16 | u16 | u16 | u16 | u32 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| u32 | u32 | u32 | u32 | u32 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| u64 | u64 | u64 | u64 | u64 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| i8 | i8 | ERR | ERR | ERR | ERR | i8 | i16 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| i16 | i16 | ERR | ERR | ERR | ERR | i16 | i16 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| i32 | i32 | ERR | ERR | ERR | ERR | i32 | i32 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| i64 | i64 | ERR | ERR | ERR | ERR | i64 | i64 | i64 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f32 | f64 | ERR | ERR | ERR | ERR |
| f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f64 | f32 | ERR | ERR | ERR |
| f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | ERR | ERR | ERR |
| bf | bf | bf | bf | bf | bf | bf | bf | bf | bf | ERR | f32 | f64 | bf | ERR | ERR | ERR |
| tf32 | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | tf32 | ERR | ERR |
| f8e4m3fn | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | f8e4m3fn | ERR |
| f8e5m2 | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | f8e5m2 |

Legend:

- b1: `bool_`
- u8: `uint8`
- u16: `uint16`
- u32: `uint32`
- u64: `uint64`
- i8: `int8`
- i16: `int16`
- i32: `int32`
- i64: `int64`
- f16: `float16`
- f32: `float32`
- f64: `float64`
- bf: `bfloat16`
- tf32: `tfloat32`
- f8e4m3fn: `float8_e4m3fn`
- f8e5m2: `float8_e5m2`
- f8e8m0fnu: `float8_e8m0fnu`
- f4e2m1fn: `float4_e2m1fn`
- ERR: Implicit promotion between these types is not supported

## Tuples

Tuples can be used in tile code. They cannot be kernel parameters.

## Rounding Modes

*class* cuda.tile.RoundingMode
:   Rounding mode for floating-point operations.

    RN *= 'nearest\_even'*
    :   Rounds the nearest (ties to even).

    RZ *= 'zero'*
    :   Round towards zero (truncate).

    RM *= 'negative\_inf'*
    :   Round towards negative infinity.

    RP *= 'positive\_inf'*
    :   Round towards positive infinity.

    FULL *= 'full'*
    :   Full precision rounding mode.

    APPROX *= 'approx'*
    :   Approximate rounding mode.

    RZI *= 'nearest\_int\_to\_zero'*
    :   Round towards zero to the nearest integer.

## Padding Modes

*class* cuda.tile.PaddingMode
:   Padding mode for load operation.

    UNDETERMINED *= 'undetermined'*
    :   The padding value is not determined.

    ZERO *= 'zero'*
    :   The padding value is zero.

    NEG\_ZERO *= 'neg\_zero'*
    :   The padding value is negative zero.

    NAN *= 'nan'*
    :   The padding value is NaN.

    POS\_INF *= 'pos\_inf'*
    :   The padding value is positive infinity.

    NEG\_INF *= 'neg\_inf'*
    :   The padding value is negative infinity.
