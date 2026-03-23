# cuTile Python Minimal Skill

Source: official NVIDIA cuTile Python docs converted into a single Markdown-like file.

Use the documented `cuda.tile` APIs via the common alias `import cuda.tile as ct`.
This file intentionally keeps the exact API spellings visible near the top.

## Exact API Names

- `ct.kernel`
- `ct.launch`
- `ct.load`
- `ct.store`
- `ct.gather`
- `ct.bid`
- `ct.sum`
- `ct.mma`
- `ct.arange`
- `ct.cdiv`
- `ct.where`
- `ct.exp`
- `ct.matmul`
- `ct.full`
- `ct.transpose`
- `ct.num_tiles`
- `ct.num_blocks`
- `ct.sqrt`
- `ct.atomic_add`
- `ct.exp2`
- `ct.max`
- `ct.log`
- `ct.tanh`
- `ct.rsqrt`
- `ct.expand_dims`
- `ct.reshape`
- `ct.broadcast_to`
- `ct.maximum`
- `ct.truediv`
- `ct.permute`
- `ct.cat`
- `ct.scatter`
- `ct.prod`
- `ct.RoundingMode.APPROX`
- `flush_to_zero`

## cuTile Python — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/

cuTile Python
#
cuTile is a parallel programming model for NVIDIA GPUs and a Python-based
DSL
.
It automatically leverages advanced hardware capabilities, such as tensor cores and tensor memory accelerators,
while providing portability across different NVIDIA GPU architectures.
cuTile enables the latest hardware features without requiring code changes.
cuTile
kernels
are GPU programs that are executed in parallel on a logical
grid
of
blocks
.
The
@ct.kernel
decorator marks a Python function as a kernel’s entry point.
Kernels cannot be called directly from the host code; the host must queue kernels for execution on GPU
using the
ct.launch()
function:
import
cuda.tile
as
ct
import
cupy
TILE_SIZE
=
16
# cuTile kernel for adding two dense vectors. It runs in parallel on the GPU.
@ct
.
kernel
def
vector_add_kernel
(
a
,
b
,
result
):
block_id
=
ct
.
bid
(
0
)
a_tile
=
ct
.
load
(
a
,
index
=
(
block_id
,),
shape
=
(
TILE_SIZE
,))
b_tile
=
ct
.
load
(
b
,
index
=
(
block_id
,),
shape
=
(
TILE_SIZE
,))
result_tile
=
a_tile
+
b_tile
ct
.
store
(
result
,
index
=
(
block_id
,),
tile
=
result_tile
)
# Host-side function that launches the above kernel.
def
vector_add
(
a
:
cupy
.
ndarray
,
b
:
cupy
.
ndarray
,
result
:
cupy
.
ndarray
):
assert
a
.
shape
==
b
.
shape
==
result
.
shape
grid
=
(
ct
.
cdiv
(
a
.
shape
[
0
],
TILE_SIZE
),
1
,
1
)
ct
.
launch
(
cupy
.
cuda
.
get_current_stream
(),
grid
,
vector_add_kernel
,
(
a
,
b
,
result
))
Kernels
move data between
arrays
and
tiles
using functions like
ct.load()
and
ct.store()
.
Both arrays and tiles are tensor-like data structures: each has a specific shape
(i.e., the number of elements along each axis) and a
dtype
(i.e., the data type of elements).
However, there are important differences:
Arrays
are stored in the global memory. They are mutable and have physical, strided
memory layouts. Within the kernel code, they support only a limited set of operations,
mostly related to
loading and storing
data to/from tiles. Various Python objects,
including PyTorch tensors and CuPy arrays, can be passed as arrays from the host code
to the kernel via kernel arguments.
Tiles
are immutable values without defined storage that only exist in the kernel code.
Tile dimensions must be compile-time constants that are powers of two.
Tiles support a multitude of
operations
, including elementwise arithmetic,
matrix multiplication, reduction, shape manipulation, etc.
Proceed to the
Quickstart
page for installation instructions and a complete working example.

## Quickstart — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/quickstart.html

Quickstart
#
This page will guide you through getting setup and running with cuTile Python, including running a first example.
Prerequisites
#
cuTile Python requires the following:
Linux x86_64, Linux aarch64 or Windows x86_64
A GPU with compute capability 8.x 10.x, 11.x or 12.x
NVIDIA Driver r580 or later
Python version 3.10, 3.11, 3.12 or 3.13
Installing cuTile Python
#
cuTile Python depends on CUDA TileIR compiler
tileiras
, which futher depends on
ptxas
and
libnvvm
from the CUDA Toolkit.
If your system does not have system-wide CUDA Toolkit (13.1+), you can install cuTile Python along with
[tileiras]
,
which installs
nvidia-cuda-tileiras
,
nvidia-cuda-nvcc
and
nvidia-nvvm
into your Python virtual environment.
pip
install
cuda-tile
[
tileiras
]
Note: the package versions for
nvidia-cuda-tileiras
,
nvidia-cuda-nvcc
and
nvidia-nvvm
must match up to the same major.minor version.
Alternatively if you already have system-wide CUDA Toolkit (13.1+) installed, you can install cuTile Python as a
standalone package. cuTile automatically searches for
tileiras
from the location of CUDA Toolkit.
pip
install
cuda-tile
Other Packages
#
Some of the cuTile Python samples also use other Python packages.
The quickstart sample on this page uses cupy, which can be installed with:
pip
install
cupy-cuda13x
The cuTile Python samples in the
samples/
directory also use pytest, torch, and numpy packages.
For PyTorch installation instructions, see
https://pytorch.org/get-started/locally/
.
Pytest and Numpy can be installed with:
pip
install
pytest
numpy
Example Code
#
The following example shows vector addition, a typical first kernel for CUDA, but uses cuTile for tile-based programming. This makes use of a 1-dimensional tile to add two 1-dimensional vectors.
This example shows a structure common to cuTile kernels:
Load one or more tiles from GPU memory
Perform computation(s) on the tile(s), resulting in new tile(s)
Write the resulting tile(s) out to GPU memory
In this case, the kernel loads tiles from two vectors,
a
and
b
. These loads create tiles called
a_tile
and
b_tile
. These tiles are added together to form a third tile, called
result
. In the last step, the kernel stores the
result
tile to the output vector
c
.
More samples can be found in the cuTile Python
repository
.
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
Example demonstrating simple vector addition.
Shows how to perform elementwise operations on vectors.
"""
import
cupy
as
cp
import
numpy
as
np
import
cuda.tile
as
ct
@ct
.
kernel
def
vector_add
(
a
,
b
,
c
,
tile_size
:
ct
.
Constant
[
int
]):
# Get the 1D pid
pid
=
ct
.
bid
(
0
)
# Load input tiles
a_tile
=
ct
.
load
(
a
,
index
=
(
pid
,),
shape
=
(
tile_size
,))
b_tile
=
ct
.
load
(
b
,
index
=
(
pid
,),
shape
=
(
tile_size
,))
# Perform elementwise addition
result
=
a_tile
+
b_tile
# Store result
ct
.
store
(
c
,
index
=
(
pid
,
),
tile
=
result
)
def
test
():
# Create input data
vector_size
=
2
**
12
tile_size
=
2
**
4
grid
=
(
ct
.
cdiv
(
vector_size
,
tile_size
),
1
,
1
)
rng
=
cp
.
random
.
default_rng
()
a
=
rng
.
random
(
vector_size
)
b
=
rng
.
random
(
vector_size
)
c
=
cp
.
zeros_like
(
a
)
# Launch kernel
ct
.
launch
(
cp
.
cuda
.
get_current_stream
(),
grid
,
# 1D grid of processors
vector_add
,
(
a
,
b
,
c
,
tile_size
))
# Copy to host only to compare
a_np
=
cp
.
asnumpy
(
a
)
b_np
=
cp
.
asnumpy
(
b
)
c_np
=
cp
.
asnumpy
(
c
)
# Verify results
expected
=
a_np
+
b_np
np
.
testing
.
assert_array_almost_equal
(
c_np
,
expected
)
print
(
"✓ vector_add_example passed!"
)
if
__name__
==
"__main__"
:
test
()
Run this from a command line as shown below. If everything has been setup correctly, the test will print that the example passed.
$
python3
samples/quickstart/VectorAdd_quickstart.py
✓
vector_add_example
passed!
To run more of the cuTile Python examples, you can directly run the samples by invoking them in the same way as the quickstart example:
$
python3
samples/FFT.py
# output not shown
You can also use pytest to run all the samples:
$
pytest
samples
=========================
test
session
starts
=========================
platform
linux
--
Python
3
.12.3,
pytest-9.0.1,
pluggy-1.6.0
rootdir:
/home/ascudiero/sw/cutile-python
configfile:
pytest.ini
collected
6
items

samples/test_samples.py
......
[
100
%
]
=========================
6
passed
in
30
.74s
==========================
Developer Tools
#
NVIDIA Nsight Compute
can profile cuTile Python kernels in the same way as SIMT CUDA kernels. With NVIDIA Nsight Compute installed, the quickstart vector addition kernel introduced here can be profiled using the following command to create a profile:
ncu
-o
VecAddProfile
--set
detailed
python3
VectorAdd_quickstart.py
This profile can then be loaded in a graphical instance of Nsight Compute and the kernel
vector_add
selected to see statistics about the kernel.
Note
Capturing detailed statistics for cuTile Python kernels requires running on NVIDIA Driver equals or later than r580.126.09 (linux) or r582.16 (windows).
On this page

## Data Model — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/data.html

Data Model
#
cuTile is an array-based programming model.
The fundamental data structure is multidimensional arrays with elements of a single homogeneous type.
cuTile Python does not expose pointers, only arrays.
An array-based model was chosen because:
Arrays know their bounds, so accesses can be checked to ensure safety and correctness.
Array-based load/store operations can be efficiently lowered to speed-of-light hardware mechanisms.
Python programmers are used to array-based programming frameworks such as NumPy.
Pointers are not a natural choice for Python.
Within
tile code
, only the types described in this section are supported.
Global Arrays
#
A
global array
(or
array
) is a container of elements of a specific
dtype
arranged in a logical multidimensional space.
Array’s
shape
is a tuple of integer values, each denoting the length of
the corresponding dimension.
The length of the shape tuple equals the arrays’s number of dimensions.
The product of shape values equals the total logical number of elements in the array.
Arrays are stored in global memory using a
strided memory layout
: in addition to a shape,
an array also has an equally sized tuple of
strides
. Strides determine the mapping of logical
array indices to physical memory locations. For example, for a 3-dimensional
float32
array
with strides
(s1, s2, s3)
, the memory address of the element at the logical index
(i1, i2, i3)
will be:
base_addr
+
4
*
(
s1
*
i1
+
s2
*
i2
+
s3
*
i3
),
where
base_addr
is the base address of the array and
4
is the byte size of a single
float32
element.
New arrays can only be allocated by the host, and passed to the tile kernel as arguments.
Tile code
can only create new views of existing arrays, for example using
Array.slice()
. Like in Python, assigning an array object to another variable does not copy
the underlying data, but creates another reference to the array object.
Any object that implements the
DLPack
interface or the
CUDA Array Interface
can be passed to the kernel as an argument. Example:
CuPy
arrays and
PyTorch
tensors.
If two or more array arguments are passed to the kernel, their memory storage must not overlap.
Otherwise, behavior is undefined.
Array’s shape can be queried using the
Array.shape
attribute, which
returns a tuple of
int32
scalars. These scalars are non-constant, runtime values.
Using
int32
makes the tile code more performant at the cost of limiting the maximum
representable shape at 2,147,483,647 elements. This limitation will be lifted in the future.
See also
cuda.tile.Array class documentation
Tiles and Scalars
#
A
tile
is an immutable multidimensional collection of elements of a specific
dtype
.
Tile’s
shape
is a tuple of integer values, each denoting the length of the corresponding dimension.
The length of the shape tuple equals the tile’s number of dimensions.
The product of shape values equals the total number of elements in the tile.
The shape of a tile must be known at compile time. Each dimension of a tile must be a power of 2.
Tile’s dtype and shape can be queried with the
dtype
and
shape
attributes, respectively.
For example, if
x
is a
float32
tile, the expression
x.dtype
will return
a compile-time constant equal to
cuda.tile.float32
.
A zero-dimensional tile is called a
scalar
. Such tile has exactly one element. The shape
of a scalar is the empty tuple
()
. Numeric literals like
7
or
3.14
are treated as
constant scalars, i.e. zero-dimensional tiles.
Since scalars are tiles, they slightly differ in behavior from Python’s
int
/
float
objects.
For example, they have
dtype
and
shape
attributes:
a
=
0
# The following line will evaluate to cuda.tile.int32 in cuTile,
# but would raise an AttributeError in Python:
a
.
dtype
Tiles can only be used in
tile code
, not host code.
The contents of a tile do not necessarily have a physical representation in memory.
Non-scalar tiles can be created by loading from
global arrays
using functions such as
cuda.tile.load()
and
cuda.tile.gather()
or with
factory
functions
such as
cuda.tile.zeros()
.
Tiles can also be stored into global arrays using functions such as
cuda.tile.store()
or
cuda.tile.scatter()
.
Only scalars (i.e. 0-dimensional tiles) can be used as
kernel
parameters.
Scalar constants are
loosely typed
by default, for example, a literal
2
or
a constant attribute like
Tile.ndim
,
Tile.shape
, or
Array.ndim
.
See also
cuda.tile.Tile class documentation
Element & Tile Space
#
The
element space
of an array is the multidimensional space of elements contained in that array,
stored in memory according to a certain layout (row major, column major, etc).
The
tile space
of an array is the multidimensional space of tiles into that array of a certain
tile shape.
A tile index
(i,
j,
...)
with shape
S
refers to the elements of the array that belong to the
(i+1)
-th,
(j+1)
-th, … tile.
When accessing the elements of an array using tile indices, the multidimensional memory layout of the array is used.
To access the tile space with a different memory layout, use the
order
parameter of load/store operations.
Shape Broadcasting
#
Shape broadcasting
allows
tiles
with different shapes to be combined in arithmetic operations.
When performing operations between
tiles
of different shapes, the smaller
tile
is automatically
extended to match the shape of the larger one, following these rules:
Tiles
are aligned by their trailing dimensions.
If the corresponding dimensions have the same size or one of them is 1, they are compatible.
If one
tile
has fewer dimensions, its shape is padded with 1s on the left.
Broadcasting follows the same semantics as
NumPy
, which makes code more concise and readable
while maintaining computational efficiency.
Data Types
#
class
cuda.tile.
DType
#
A
data type
(or
dtype
) describes the type of the objects of an
array
,
tile
, or
operation.
Dtypes
determine how values are stored in memory and how operations on those values are
performed.
Dtypes
are immutable.
Dtypes
can be used in
host code
and
tile code
.
They can be
kernel
parameters.
property
bitwidth
#
The number of bits in an element of the
data type
.
property
name
#
The name of the
data type
.
cuda.tile.
bool_
#
A 8-bit
arithmetic dtype
(
True
or
False
).
cuda.tile.
uint8
#
A 8-bit unsigned integer
arithmetic dtype
whose values exist on the interval [0, +256].
cuda.tile.
uint16
#
A 16-bit unsigned integer
arithmetic dtype
whose values exist on the interval [0, +65,536].
cuda.tile.
uint32
#
A 32-bit unsigned integer
arithmetic dtype
whose values exist on the interval [0, +4,294,967,295].
cuda.tile.
uint64
#
A 64-bit unsigned integer
arithmetic dtype
whose values exist on the interval [0, +18,446,744,073,709,551,615].
cuda.tile.
int8
#
A 8-bit signed integer
arithmetic dtype
whose values exist on the interval [−128, +127].
cuda.tile.
int16
#
A 16-bit signed integer
arithmetic dtype
whose values exist on the interval [−32,768, +32,767].
cuda.tile.
int32
#
A 32-bit signed integer
arithmetic dtype
whose values exist on the interval [−2,147,483,648, +2,147,483,647].
cuda.tile.
int64
#
A 64-bit signed integer
arithmetic dtype
whose values exist on the interval [−9,223,372,036,854,775,808, +9,223,372,036,854,775,807].
cuda.tile.
float16
#
A IEEE 754 half-precision (16-bit) binary floating-point
arithmetic dtype
(see
IEEE 754-2019
).
cuda.tile.
float32
#
A IEEE 754 single-precision (32-bit) binary floating-point
arithmetic dtype
(see
IEEE 754-2019
).
cuda.tile.
float64
#
A IEEE 754 double-precision (64-bit) binary floating-point
arithmetic dtype
(see
IEEE 754-2019
).
cuda.tile.
bfloat16
#
A 16-bit floating-point
arithmetic dtype
with 1 sign bit, 8 exponent bits, and 7 mantissa bits.
cuda.tile.
tfloat32
#
A 32-bit tensor floating-point
numeric dtype
with 1 sign bit, 8 exponent bits, and 10 mantissa bits (19-bit representation stored in 32-bit container).
cuda.tile.
float8_e4m3fn
#
An 8-bit floating-point
numeric dtype
with 1 sign bit, 4 exponent bits, and 3 mantissa bits.
cuda.tile.
float8_e5m2
#
An 8-bit floating-point
numeric dtype
with 1 sign bit, 5 exponent bits, and 2 mantissa bits.
Numeric & Arithmetic Data Types
#
A
numeric
data type represents numbers. An
arithmetic
data type is a numeric data type
that supports general arithmetic operations such as addition, subtraction, multiplication,
and division.
Arithmetic Promotion
#
Binary operations can be performed on two
tile
or
scalar
operands of different
numeric dtypes
.
When both operands are
loosely typed numeric constants
, then the result is also
a loosely typed constant: for example,
5
+
7
is a loosely typed integral constant 12,
and
5
+
3.0
is a loosely typed floating-point constant 8.0.
If any of the operands is not a
loosely typed numeric constant
, then both are
promoted
to a common dtype using the following process:
Each operand is classified into one of the three categories:
boolean
,
integral
, or
floating-point
.
The categories are ordered as follows:
boolean
<
integral
<
floating-point
.
If either operand is a
loosely typed numeric constant
, a concrete dtype is picked for it:
integral constants are treated as
int32
,
int64
, or
uint64
, depending on the value;
floating-point constants are treated as
float32
.
If one of the two operands has a higher category than the other, then its concrete dtype
is chosen as the common dtype.
If both operands are of the same category, but one of them is a
loosely typed numeric constant
,
then the other operand’s dtype is picked as the common dtype.
Otherwise, the common dtype is computed according to the table below.
b1
u8
u16
u32
u64
i8
i16
i32
i64
f16
f32
f64
bf
tf32
f8e4m3fn
f8e5m2
b1
b1
u8
u16
u32
u64
i8
i16
i32
i64
f16
f32
f64
bf
ERR
ERR
ERR
u8
u8
u8
u16
u32
u64
ERR
ERR
ERR
ERR
f16
f32
f64
bf
ERR
ERR
ERR
u16
u16
u16
u16
u32
u64
ERR
ERR
ERR
ERR
f16
f32
f64
bf
ERR
ERR
ERR
u32
u32
u32
u32
u32
u64
ERR
ERR
ERR
ERR
f16
f32
f64
bf
ERR
ERR
ERR
u64
u64
u64
u64
u64
u64
ERR
ERR
ERR
ERR
f16
f32
f64
bf
ERR
ERR
ERR
i8
i8
ERR
ERR
ERR
ERR
i8
i16
i32
i64
f16
f32
f64
bf
ERR
ERR
ERR
i16
i16
ERR
ERR
ERR
ERR
i16
i16
i32
i64
f16
f32
f64
bf
ERR
ERR
ERR
i32
i32
ERR
ERR
ERR
ERR
i32
i32
i32
i64
f16
f32
f64
bf
ERR
ERR
ERR
i64
i64
ERR
ERR
ERR
ERR
i64
i64
i64
i64
f16
f32
f64
bf
ERR
ERR
ERR
f16
f16
f16
f16
f16
f16
f16
f16
f16
f16
f16
f32
f64
ERR
ERR
ERR
ERR
f32
f32
f32
f32
f32
f32
f32
f32
f32
f32
f32
f32
f64
f32
ERR
ERR
ERR
f64
f64
f64
f64
f64
f64
f64
f64
f64
f64
f64
f64
f64
f64
ERR
ERR
ERR
bf
bf
bf
bf
bf
bf
bf
bf
bf
bf
ERR
f32
f64
bf
ERR
ERR
ERR
tf32
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
tf32
ERR
ERR
f8e4m3fn
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
f8e4m3fn
ERR
f8e5m2
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
ERR
f8e5m2
Legend:
b1:
bool_
u8:
uint8
u16:
uint16
u32:
uint32
u64:
uint64
i8:
int8
i16:
int16
i32:
int32
i64:
int64
f16:
float16
f32:
float32
f64:
float64
bf:
bfloat16
tf32:
tfloat32
f8e4m3fn:
float8_e4m3fn
f8e5m2:
float8_e5m2
f8e8m0fnu:
float8_e8m0fnu
f4e2m1fn:
float4_e2m1fn
ERR: Implicit promotion between these types is not supported
Tuples
#
Tuples can be used in
tile code
. They cannot be
kernel
parameters.
Rounding Modes
#
class
cuda.tile.
RoundingMode
#
Rounding mode for floating-point operations.
RN
=
'nearest_even'
#
Rounds the nearest (ties to even).
RZ
=
'zero'
#
Round towards zero (truncate).
RM
=
'negative_inf'
#
Round towards negative infinity.
RP
=
'positive_inf'
#
Round towards positive infinity.
FULL
=
'full'
#
Full precision rounding mode.
APPROX
=
'approx'
#
Approximate rounding mode.
RZI
=
'nearest_int_to_zero'
#
Round towards zero to the nearest integer.
Padding Modes
#
class
cuda.tile.
PaddingMode
#
Padding mode for load operation.
UNDETERMINED
=
'undetermined'
#
The padding value is not determined.
ZERO
=
'zero'
#
The padding value is zero.
NEG_ZERO
=
'neg_zero'
#
The padding value is negative zero.
NAN
=
'nan'
#
The padding value is NaN.
POS_INF
=
'pos_inf'
#
The padding value is positive infinity.
NEG_INF
=
'neg_inf'
#
The padding value is negative infinity.
On this page

## Execution Model — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/execution.html

Execution Model
#
Abstract Machine
#
A
tile kernel
is executed by logical thread
blocks
that are organized in
a 1D, 2D, or 3D
grid
.
Each
block
is executed by a subset of a GPU, which is decided by the
implementation, not the programmer.
Each
block
executes the body of the
kernel
.
Scalar operations are executed serially by a single thread of the
block
,
and array operations are collectively executed in parallel by all threads of
the
block
.
Tile programs explicitly describe
block
-level parallelism, but not
thread-level parallelism.
Threads cannot be explicitly identified or manipulated in tile programs.
Explicit synchronization or communication within a
block
is not
permitted, but it is allowed between different
blocks
.
It is important to not confuse
blocks
(units of execution) with
tiles
(units of data).
A block may work with multiple different
tiles
with
differing shapes originating from differing
global arrays
.
Execution Spaces
#
cuTile code is executed on one or more
targets
, which are distinct execution environments that
are distinguished by different hardware resources or programming models.
A function is
usable
if it can be called.
A type or object is
usable
if its attributes are accessible (can be read and written) and its
methods are callable.
Some functions, types, and objects are only usable on certain
targets
.
The set of
targets
that such a construct is usable on is called its
execution space
.
Host code
is the execution space that includes all CPU targets.
SIMT code
is the execution space that includes all CUDA SIMT targets.
Note: This has historically been called device code, but we avoid this term to prevent ambiguity.
Tile code
is the execution space that includes all CUDA tile targets.
Functions can have decorators that explicitly specify their execution space.
These are called
annotated functions
.
Tile Functions
#
class
cuda.tile.
function
(
func
=
None
,
/
,
*
,
host
=
False
,
tile
=
True
)
#
Tile functions
are functions that are usable in
tile code
.
This decorator indicates what
execution spaces
a function can be called from.
With no arguments, it denotes a tile-only function.
When an unannotated function is called by a
tile function
, tile shall be added to the
unannotated function’s execution space.
This process is recursive.
No explicit annotation is required.
The types usable as parameters to a
tile function
are described in the
data model
.
Parameters
:
host
(
bool
,
optional
) – Whether the function can be called from
host code
.
Default is False.
tile
(
bool
,
optional
) – Whether the function can be called from
tile code
.
Default is True.
Tile Kernels
#
class
cuda.tile.
kernel
(
function
=
None
,
/
,
**
kwargs
)
#
A
tile kernel
is a function executed by each
block
in a
grid
.
Functions with this decorator are
kernels
.
Kernels
are the entry points of
tile code
.
Their
execution space
shall be only
tile code
; they cannot be called from
host code
.
Kernels cannot be called directly. Instead, use
launch()
to
queue a kernel for execution over a grid.
The types usable as parameters to a
kernel
are described in the
data model
.
Parameters
:
num_ctas
– Number of CTAs in a CGA. Must be a power of 2 between 1 and 16, inclusive.
Default: None (auto).
occupancy
– Expected number of active CTAs per SM, [1, 32]. Default: None (auto).
opt_level
– Optimization level [0, 3], default 3.
Target-specific values for the compiler options above can be provided
using a
ByTarget
object.
Examples:
@ct
.
kernel
def
f
(
a
,
b
,
c
):
pass
grid
=
(
8
,
8
)
ct
.
launch
(
stream
,
grid
,
f
,
(
A
,
B
,
C
))
cuda.tile.
launch
(
stream
,
grid
,
kernel
,
kernel_args
,
/
)
#
Launch a cuTile kernel.
Parameters
:
stream
– The CUDA stream to execute the
kernel
on.
grid
– Tuple of up to 3 grid dimensions to execute the
kernel
over.
kernel
– The
kernel
to execute.
kernel_args
– Positional arguments to pass to the kernel.
Python Subset
#
Tile code
supports a subset of the Python language.
Within
tile code
, there is no Python runtime.
Only Python features explicitly enumerated in this document are supported.
Many features, such as lambdas, exceptions, and coroutines are not supported today.
Object Model & Lifetimes
#
All objects created within
tile code
are immutable.
Any operation that conceptually modifies an object or its attributes creates and returns a new
object.
Attributes cannot be dynamically added to objects.
The only mutable objects that can be used in
tile code
are
arrays
, which must be passed in as
kernel
parameters.
The caller of a
kernel
must ensure that:
No
arrays
passed to the
kernel
alias one another.
All passed
arrays
remain valid until the
kernel
has finished execution.
Control Flow
#
Python control flow statements (
if
,
for
,
while
, etc.) shall be usable in
tile code
.
They can be arbitrarily nested.
Current limitations
#
The Python subset used in
tile code
imposes additional restrictions on control flow:
step
must be strictly positive.
Negative-step ranges such as
range(10,
0,
-1)
are not supported today. Passing a negative step
indirectly via a variable may lead to undefined behavior.
Tile Parallelism
#
When a
block
executes a function that takes
tiles
as parameters, it may parallelize the
evaluation of the function across the
block
’s execution resources.
Unless otherwise specified, the execution shall complete before the function returns.
Constantness
#
Constant Expressions & Objects
#
Some facilities require certain parameters to be an object that is known statically at compilation time.
Constant expressions
produce
constant objects
suitable for such parameters. Constant expressions are:
A literal object.
Integer arithmetic expressions where all the operands are literal objects.
A local object or parameter that is assigned from a literal object or constant expression.
A global object that is defined at the time of compilation or launch.
By default, numeric constants are
loosely typed
: until used in a context that requires
a type of a specific width, integer constants have infinite precision, and floating-point
constants are stored in the IEEE 754 double precision format.
A
strictly typed
constant can be created by calling a dtype object as a constructor,
e.g.
ct.int16(5)
creates a strictly typed
int16
constant. When a strictly typed constant
is combined with a loosely typed constant, the result is a strictly typed constant.
For example
ct.int16(5)
+
2
will create a strictly typed
int16
constant 7.
Combining two strictly typed constants creates a new strictly typed constant. In this case,
the regular
type promotion
rules apply.
For example,
ct.int16(5)
+
ct.int32(7)
will create a strictly typed
int32
constant 12.
Constant Embedding
#
If a parameter to a
kernel
is
constant embedded
, then:
All uses of the parameter shall act as if they were replaced by the literal value of the parameter.
There shall be a distinct machine representation of the
kernel
for each different value of the parameter that the
kernel
is invoked with. Note: The
kernel
shall be compiled once for each different value of the parameter, even if JIT caching is enabled.
The
machine representation
of the parameter shall be 0 bytes.
Constant Type Hints
#
import
cuda.tile
as
ct
def
needs_constant
(
x
:
ct
.
Constant
):
pass
def
needs_constant_int
(
x
:
ct
.
Constant
[
int
]):
pass
class
cuda.tile.
ConstantAnnotation
#
A
typing.Annotated
metadata class indicating that an object shall be
constant embedded
.
If an object of this class is passed as a metadata argument to a
typing.Annotated
type hint
on a parameter, then the parameter shall be a constant embedded.
cuda.tile.
Constant
#
A type hint indicating that a value shall be
constant embedded
.
It can be used either with (
Constant[int]
) or without (
Constant
, meaning a constant of any
type) an underlying type hint.
alias of
Annotated
[
T
, ConstantAnnotation()]
On this page

## Debugging — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/debugging.html

Debugging
#
Exception Types
#
class
cuda.tile.
TileSyntaxError
#
Exception when a python syntax not supported by cuTile is encountered.
class
cuda.tile.
TileTypeError
#
Exception when an unexpected type or
data type
is encountered.
class
cuda.tile.
TileValueError
#
Exception when an unexpected python value is encountered.
class
cuda.tile.
TileUnsupportedFeatureError
#
Exception when a feature is not supported by the underlying compiler or
the GPU architecture.
class
cuda.tile.
TileCompilerExecutionError
#
Exception when
tileiras
compiler throws an error.
class
cuda.tile.
TileCompilerTimeoutError
#
Exception when
tileiras
compiler timeout limit is exceeded.
Environment Variables
#
The following environment variables are useful when
the above exceptions are encountered during kernel
development.
Set
CUDA_TILE_ENABLE_CRASH_DUMP=1
to enable dumping
an archive including the TileIR bytecode
for submitting a bug report on
TileCompilerExecutionError
or
TileCompilerTimeoutError
.
Set
CUDA_TILE_COMPILER_TIMEOUT_SEC
to limit the
time the TileIR compiler
tileiras
can take.
Set
CUDA_TILE_LOGS=CUTILEIR
to print cuTile Python
IR during compilation to stderr. This is useful when
debugging
TileTypeError
.
Set
CUDA_TILE_TEMP_DIR
to configure the directory
for storing temporary files.
Set
CUDA_TILE_CACHE_DIR
to configure the directory
for the bytecode-to-cubin disk cache. Compiled cubins
are cached here to avoid recompilation of unchanged
kernels. Set to
0
,
off
,
none
, or an empty
string to disable caching. Defaults to
~/.cache/cutile-python
.
Set
CUDA_TILE_CACHE_SIZE
to configure the maximum
disk cache size in bytes. Oldest entries are evicted
when the cache exceeds this limit. Defaults to
2 GB (2147483648).
On this page

## cuda.tile.load — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.load.html

cuda.tile.load
#
cuda.tile.
load
(
array
,
/
,
index
,
shape
,
*
,
order
=
'C'
,
padding_mode
=
PaddingMode.UNDETERMINED
,
latency
=
None
,
allow_tma
=
None
,
)
#
Loads a tile from the
array
which is partitioned into a
tile space
.
The
tile space
is the result of partitioning the
array
into a grid of equally
sized tiles specified by
shape
.
For example, partitoning a 2D
array
of shape
(M,
N)
using tile shape
(tm,
tn)
results in a 2D tile space of size
(cdiv(M,
tm),
cdiv(N,
tn))
.
An index into this tile space using index
(i,
j)
produces a tile of size
(tm,
tn)
:
>>>
t
=
ct
.
load
(
array
,
(
i
,
j
),
(
tm
,
tn
))
# `t` has shape (tm, tn)
The result tile
t
will be computed according to
t
[
x
,
y
]
=
array
[
i
*
tm
+
x
,
j
*
tn
+
y
]
(
for
all
0
<=
x
<
tm
,
0
<=
y
<
tn
)
For access that is out of bound, the value will be determined by
padding_mode
.
order
is used to map the tile axis to the array axis. The transposed example of the above call
to
load
would be:
>>>
ct
.
load
(
array
,
(
j
,
i
),
shape
=
(
tn
,
tm
),
order
=
(
1
,
0
))
The result tile
t
will be computed according to
t
[
y
,
x
]
=
array
[
i
*
tm
+
x
,
j
*
tn
+
y
]
Parameters
:
array
(
Array
) – The
array
to load from.
index
(
tuple
[
int
,
...
]
) – An index in the
tile space
of
shape
from
array
.
shape
(
tuple
[
const int
,
...
]
) – A tuple of const integers definining the shape of the tile.
order
(
"C"
or
"F"
, or
tuple
[
const int
,
...
]
) –
Permutation applied to array axes before the
logical
tile space
is constructed. Can be specified either as a tuple of constants,
or as one of the two special string literal values:
”C” is an alias for
(0,
1,
2,
...)
, i.e. no permutation applied;
”F” is an alias for
(...,
2,
1,
0)
, i.e. axis order is reversed.
padding_mode
(
PaddingMode
) – The value used to pad the tile when it extends beyond the array
boundaries. By default, the padding value is undetermined.
latency
(
const int
) – A hint indicating how heavy DRAM traffic will be. It shall be an
integer between 1 (low) and 10 (high). By default, the compiler will infer the latency.
allow_tma
(
const bool
) – If False, the load will not use TMA. By default, TMA is allowed.
Return type
:
Tile
Examples
>>>
# Regular load.
>>>
tile
=
ct
.
load
(
array2d
,
(
0
,
0
),
shape
=
(
2
,
4
))
>>>
# Load with a transpose.
>>>
tile
=
ct
.
load
(
array2d
,
(
0
,
0
),
shape
=
(
4
,
2
),
order
=
'F'
)
>>>
# Load transposing the last two axes.
>>>
tile
=
ct
.
load
(
array3d
,
(
0
,
0
,
0
),
shape
=
(
8
,
4
,
2
),
order
=
(
0
,
2
,
1
))
>>>
# Load a single element as 0d tile
>>>
tile
=
ct
.
load
(
array3d
,
(
0
,
0
,
0
),
shape
=
())
See also
store()
gather()
Tile space

## cuda.tile.store — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.store.html

cuda.tile.store
#
cuda.tile.
store
(
array
,
/
,
index
,
tile
,
*
,
order
=
'C'
,
latency
=
None
,
allow_tma
=
None
,
)
#
Stores a
tile
value into the
array
at the
index
of its
tile space
.
The
tile space
is the result of partitioning the
array
into a grid of tiles
with equal size defined by the shape of the
tile
.
For example, given a tile
t
of shape
(tm,
tn)
and array of shape
(M,
N)
:
>>>
# tile `t` has shape (tm, tn)
>>>
ct
.
store
(
array
,
(
i
,
j
),
t
)
The above call to
store
will store elements according to:
array
[
i
*
tm
+
x
,
i
*
tn
+
y
]
=
t
[
x
,
y
]
(
for
0
<=
x
<
tm
,
0
<=
y
<
tn
)
Access which falls out of the boundary of the
array
will be ignored.
Parameters
:
array
(
Array
) – The
array
to store to.
index
(
tuple
[
int
,
...
]
) – An index in the
tile space
of
array
.
shape
is inferred from the
tile
argument.
tile
(
Tile
) – The
tile
to store. The rank of the tile must match rank of the array,
unless it is a scalar or 0d tile.
order
(
"C"
or
"F"
, or
tuple
[
const int
,
...
]
) – Order of axis mapping. See
load()
.
latency
(
int
,
optional
) – A hint indicating how heavy DRAM traffic will be. It shall be an
integer between 1 (low) and 10 (high). By default, the compiler will infer the latency.
allow_tma
(
bool
,
optional
) – If False, the load will not use TMA. By default, TMA is allowed.
Examples
>>>
tile
=
ct
.
load
(
array_in
,
bid_x
,
shape
=
4
)
>>>
tile
=
tile
*
2
>>>
ct
.
store
(
array_out
,
(
bid_x
,),
tile
=
tile
)
# store a scalar
>>>
ct
.
store
(
array_out
,
(
0
,),
tile
=
0
)
See also
load()
scatter()
Tile space

## cuda.tile.gather — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.gather.html

cuda.tile.gather
#
cuda.tile.
gather
(
array
,
indices
,
/
,
*
,
mask
=
None
,
padding_value
=
0
,
check_bounds
=
True
,
latency
=
None
,
)
#
Loads a tile from the
array
elements specified by
indices
.
indices
must be a tuple whose length equals the
array
rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
The result shape will be the same as the broadcasted shape of indices.
For example, consider a 2-dimensional array. In this case, indices must be a tuple
of length 2. Suppose that
ind0
and
ind1
are integer tiles
of shapes
(M,
N,
1)
and
(M,
1,
K)
.
Then the result tile will have the broadcasted shape
(M,
N,
K)
:
>>>
t
=
ct
.
gather
(
array
,
(
ind0
,
ind1
))
# `t` has shape (M, N, K)
The result tile
t
will be computed according to
t
[
i
,
j
,
k
]
=
array
[
ind0
[
i
,
j
,
0
],
ind1
[
i
,
0
,
k
]]
(
for
all
0
<=
i
<
M
,
0
<=
j
<
N
,
0
<=
k
<
K
)
If the array is 1-dimensional,
indices
can be passed as a tile rather than a tuple.
This is a convenience notation that is strictly equivalent to passing a tuple of length 1:
>>>
ct
.
gather
(
array
,
ind0
)
# equivalent to ct.gather(array, (ind0,))
A custom boolean
mask
can be provided to control which elements are loaded.
The mask must be a scalar or a tile whose shape is broadcastable to the common shape
of indices. Where the mask is
False
,
padding_value
is returned instead of loading
from the array.
gather()
checks that indices are within the bounds of the array. For indices
that are out of bounds,
padding_value
will be returned (zero by default).
It must be a scalar or a tile whose shape is broadcastable to the common shape of indices.
If both
mask
and
check_bounds=True
are provided, the effective mask is the logical
AND of both the custom mask and the bounds-checking mask. This means an element is only
loaded if both the custom mask is
True
AND the index is within bounds.
To disable bounds checking, set
check_bounds
to
False
.
In this mode, the caller is responsible for ensuring that all indices are within the bounds
of the array, and any out-of-bounds access will result in undefined behavior.
Negative indices are interpreted as out of bounds, i.e. they don’t follow the Python’s
negative index convention.

## cuda.tile.bid — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.bid.html

cuda.tile.bid
#
cuda.tile.
bid
(
axis
)
#
Gets the index of current block.
Parameters
:
axis
(
const int
) – The axis of the block index space. Possible values are 0, 1, 2.
Return type
:
int32
Examples
>>>
bid_x
=
ct
.
bid
(
0
)
>>>
bid_y
=
ct
.
bid
(
1
)
>>>
bid_z
=
ct
.
bid
(
2
)

## cuda.tile.sum — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.sum.html

cuda.tile.sum
#
cuda.tile.
sum
(
x
,
/
,
axis
=
None
,
*
,
keepdims
=
False
,
rounding_mode
=
None
,
flush_to_zero
=
False
,
)
#
Performs sum reduction on tile along the
axis
.
Parameters
:
x
(
Tile
) – input tile.
axis
(
None
|
const int
|
tuple
[
const int
,
...
]
) – the axis for reduction.
The default,
axis=None
, will reduce all of the elements.
keep_dims
(
const bool
) – If true, preserves the number of dimension
from the input tile.
rounding_mode
(
RoundingMode
) – The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
flush_to_zero
(
const bool
) – If True, flushes subnormal inputs and results to sign-preserving zero, default is False.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
2
,
4
),
3
,
dtype
=
ct
.
float32
)
>>>
ty
=
ct
.
sum
(
tx
,
1
)
>>>
ty
.
shape
(2,)
>>>
ty
=
ct
.
sum
(
tx
,
1
,
keepdims
=
True
)
>>>
ty
.
shape
(2, 1)

## cuda.tile.mma — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.mma.html

cuda.tile.mma
#
cuda.tile.
mma
(
x
,
y
,
/
,
acc
)
#
Matrix multiply-accumulate.
Computes
(x
@
y)
+
acc
as a single operation
(where
@
denotes matrix multiplication).
Preserves the dtype of
acc
.
Parameters
:
x
(
Tile
) – LHS of the mma, 2D or 3D.
y
(
Tile
) – RHS of the mma, 2D or 3D.
acc
(
Tile
) – Accumulator of mma.
Supported datatypes:
Input
Acc/Output
f16
f16 or f32
bf16
f32
f32
f32
f64
f64
tf32
f32
f8e4m3fn
f16 or f32
f8e5m2
f16 or f32
[u|i]8
i32
If
x
and
y
have different dtype, they will NOT be promoted to common dtype.
Shape of
x
and
y
will be broadcasted to up until the last two axes.
Return type
:
Tile
Example
>>>
tx
=
ct
.
full
((
2
,
4
),
3
,
dtype
=
ct
.
float32
)
>>>
ty
=
ct
.
full
((
4
,
8
),
4
,
dtype
=
ct
.
float32
)
>>>
acc
=
ct
.
full
((
2
,
8
),
0
,
dtype
=
ct
.
float32
)
# default
>>>
tz
=
ct
.
mma
(
tx
,
ty
,
acc
)

## cuda.tile.arange — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.arange.html

cuda.tile.arange
#
cuda.tile.
arange
(
size
,
/
,
*
,
dtype
)
#
Creates a tile with value starting from 0 to
size - 1
.
Parameters
:
size
(
const int
) – Size of the tile.
dtype
(
DType
) – Datatype of the tile.
Return type
:
Tile
Examples
>>>
tile
=
ct
.
arange
(
16
,
dtype
=
ct
.
int32
)

## cuda.tile.cdiv — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.cdiv.html

cuda.tile.cdiv
#
cuda.tile.
cdiv
(
x
,
y
,
/
)
#
Computes ceil(x / y). Can be used on the host.
Parameters
:
x
(
Tile
) – int tile.
y
(
Tile
) – int tile.
Return type
:
Tile
Examples
>>>
tile
=
ct
.
full
((
2
,
2
),
7
,
dtype
=
ct
.
int32
)
>>>
ct
.
cdiv
(
tile
,
4
)
Tile((2,2), dtype=int32)
>>>
ct
.
cdiv
(
7
,
4
)
2

## cuda.tile.where — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.where.html

cuda.tile.where
#
cuda.tile.
where
(
cond
,
x
,
y
,
/
)
#
Returns elements chosen from x or y depending on condition.
Parameters
:
cond
(
Tile
) – Boolean tile of shape
S
.
x
(
Tile
) – Tile of shape
S
and dtype
T
, selected if
cond
is True.
y
(
Tile
) – Tile of shape
S
and dtype
T
, selected if
cond
is False.
Return type
:
Tile
Examples
>>>
cond
=
ct
.
arange
(
4
,
dtype
=
ct
.
int32
)
>>>
cond
=
cond
>
2
>>>
x_true
=
ct
.
full
((
4
,),
1.0
,
dtype
=
ct
.
float32
)
>>>
x_false
=
ct
.
full
((
4
,),
-
1.0
,
dtype
=
ct
.
float32
)
>>>
y
=
ct
.
where
(
cond
,
x_true
,
x_false
)
>>>
y
[1., 1., -1., -1.]
>>>
z
=
ct
.
where
(
cond
,
1.0
,
-
1.0
)
>>>
z
[1., 1., -1., -1.]

## cuda.tile.exp — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.exp.html

cuda.tile.exp
#
cuda.tile.
exp
(
x
,
/
)
#
Perform
exp
on a tile.
Parameters
:
x
(
Tile
)
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
32
,
32
),
3.0
,
dtype
=
ct
.
float32
)
>>>
tx
=
ct
.
exp
(
tx
)

## cuda.tile.matmul — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.matmul.html

cuda.tile.matmul
#
cuda.tile.
matmul
(
x
,
y
,
/
)
#
Performs matrix multiply on the given tiles.
Parameters
:
x
(
Tile
) – LHS of the matmul, 1D, 2D, or 3D.
y
(
Tile
) – RHS of the matmul, 1D, 2D, or 3D.
Supported input datatypes: [f16, bf16, f32, f64, tf32, f8e4m3fn, f8e5m2, i8, u8]
If
x
and
y
have different dtype, they will first be promoted to common
dtype. The result dtype is the same as the promoted input types.
Shape of
x
and
y
will be broadcasted to up until the last two axes.
Return type
:
Tile
Example
>>>
tx
=
ct
.
full
((
2
,
4
),
3
,
dtype
=
ct
.
float32
)
>>>
ty
=
ct
.
full
((
4
,
8
),
4
,
dtype
=
ct
.
float32
)
# default
>>>
tz
=
ct
.
matmul
(
tx
,
ty
)
# use builtin `@`
>>>
tz
=
tx
@
ty

## cuda.tile.full — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.full.html

cuda.tile.full
#
cuda.tile.
full
(
shape
,
fill_value
,
dtype
)
#
Creates a tile filled with given value.
Parameters
:
shape
(
tuple
[
const int
,
...
]
) – The shape of the tile.
fill_value
(
int
|
float
|
bool
]
) – Value for the tile.
dtype
(
DType
) – The
Data type
of the tile.
Return type
:
Tile
Examples
>>>
tile
=
ct
.
full
((
4
,
4
),
3.14
,
dtype
=
ct
.
float32
)

## cuda.tile.transpose — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.transpose.html

cuda.tile.transpose
#
cuda.tile.
transpose
(
x
,
/
,
axis0
=
None
,
axis1
=
None
)
#
Transposes two axes of the input tile with at least 2 dimensions.
For a 2-dimensional tile, the two axes are transposed if
axis0
and
axis1
are not specified.
For tiles with more than 2 dimensions,
axis0
and
axis1
must be explicitly specified.
Parameters
:
x
(
Tile
) – input tile.
axis0
(
const int
) – the first axis to transpose.
axis1
(
const int
) – the second axis to transpose.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
2
,
4
,
8
),
0.
,
dtype
=
ct
.
float32
)
>>>
ty
=
ct
.
transpose
(
tx
,
axis0
=
0
,
axis1
=
1
)
>>>
ty
.
shape
(4, 2, 8)
>>>
tx
=
ct
.
full
((
2
,
4
),
0.
,
dtype
=
ct
.
float32
)
>>>
ty
=
ct
.
transpose
(
tx
)
>>>
ty
.
shape
(4, 2)

## cuda.tile.num_tiles — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.num_tiles.html

cuda.tile.num_tiles
#
cuda.tile.
num_tiles
(
array
,
/
,
axis
,
shape
,
order
=
'C'
)
#
Gets the number of tiles in the
tile space
of the array along the
axis
.
Parameters
:
array
(
Array
) – An array object on a cuda device.
axis
(
const int
) – The axis of the tile partition space to get the dim size.
shape
(
const int
...
) – A sequence of const integers definining the shape of the tile.
order
(
"C"
or
"F"
, or
tuple
[
const int
,
...
]
) – Order of axis mapping. See
load()
.
Returns
:
int32
Examples
Suppose array size is (32, 16), tile shape (4, 8),
the partition space will be (cdiv(32, 4), cdiv(16, 8)) == (8, 2)
>>>
ct
.
num_tiles
(
array
,
0
,
shape
=
(
4
,
8
))
8
>>>
ct
.
num_tiles
(
array
,
1
,
shape
=
(
4
,
8
))
2

## cuda.tile.num_blocks — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.num_blocks.html

cuda.tile.num_blocks
#
cuda.tile.
num_blocks
(
axis
)
#
Gets the number of blocks along the axis.
Parameters
:
axis
(
const int
) – The axis of the block index space. Possible values are 0, 1, 2.
Return type
:
int32
Examples
>>>
num_blocks_x
=
ct
.
num_blocks
(
0
)
>>>
num_blocks_y
=
ct
.
num_blocks
(
1
)
>>>
num_blocks_z
=
ct
.
num_blocks
(
2
)

## cuda.tile.sqrt — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.sqrt.html

cuda.tile.sqrt
#
cuda.tile.
sqrt
(
x
,
/
,
*
,
rounding_mode
=
None
,
flush_to_zero
=
False
)
#
Perform
sqrt
on a tile.
Parameters
:
x
(
Tile
)
rounding_mode
(
RoundingMode
) – The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
flush_to_zero
(
const bool
) – If True, flushes subnormal inputs and results to sign-preserving zero, default is False.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
32
,
32
),
3.0
,
dtype
=
ct
.
float32
)
>>>
tx
=
ct
.
sqrt
(
tx
)

## cuda.tile.atomic_add — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.atomic_add.html

cuda.tile.atomic_add
#
cuda.tile.
atomic_add
(
array
,
indices
,
update
,
/
,
*
,
check_bounds
=
True
,
memory_order
=
MemoryOrder.ACQ_REL
,
memory_scope
=
MemoryScope.DEVICE
,
)
#
Bulk atomic post-increment of array elements at given indices.
For each specified index,
atomic_add()
reads the corresponding array element,
adds
update
to it, and writes the modified value back to the same location.
The original value of the element before the update is returned.
For each individual element, the operation is performed atomically,
but the operation as a whole is not atomic, and the order of individual writes is unspecified.
atomic_add()
follows the same convention as
gather()
and
scatter()
:
indices
must be a tuple whose length equals the
array
rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
If the array is 1-dimensional,
indices
can be passed as a single tile
rather than a tuple of length 1.
update
must be a scalar or a tile whose shape is broadcastable to the
common shape of
indices
.
By default,
atomic_add()
checks that indices are within the bounds of the array.
For indices that are out of bounds, no operation is performed, and an implementation-defined
value is returned. To disable bounds checking, set
check_bounds
to
False
.
In this mode, the caller is responsible for ensuring that all indices are within
the bounds of the array, and any out-of-bounds access will result in undefined behavior.
Examples
>>>
indices
=
ct
.
arange
(
32
,
dtype
=
ct
.
int32
)
>>>
update
=
ct
.
arange
(
32
,
dtype
=
ct
.
int32
)
>>>
old_value
=
ct
.
atomic_add
(
array
,
indices
,
update
)

## cuda.tile.exp2 — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.exp2.html

cuda.tile.exp2
#
cuda.tile.
exp2
(
x
,
/
,
*
,
flush_to_zero
=
False
)
#
Perform
exp2
on a tile.
Parameters
:
x
(
Tile
)
flush_to_zero
(
const bool
) – If True, flushes subnormal inputs and results to sign-preserving zero, default is False.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
32
,
32
),
3.0
,
dtype
=
ct
.
float32
)
>>>
tx
=
ct
.
exp2
(
tx
)

## cuda.tile.max — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.max.html

cuda.tile.max
#
cuda.tile.
max
(
x
,
/
,
axis
=
None
,
*
,
keepdims
=
False
,
flush_to_zero
=
False
)
#
Performs max reduction on tile along the
axis
.
Parameters
:
x
(
Tile
) – input tile.
axis
(
None
|
const int
|
tuple
[
const int
,
...
]
) – the axis for reduction.
The default,
axis=None
, will reduce all of the elements.
keep_dims
(
const bool
) – If true, preserves the number of dimension
from the input tile.
flush_to_zero
(
const bool
) – If True, flushes subnormal inputs and results to sign-preserving zero, default is False.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
2
,
4
),
3
,
dtype
=
ct
.
float32
)
>>>
ty
=
ct
.
max
(
tx
,
1
)
>>>
ty
.
shape
(2,)
>>>
ty
=
ct
.
max
(
tx
,
1
,
keepdims
=
True
)
>>>
ty
.
shape
(2, 1)

## cuda.tile.log — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.log.html

cuda.tile.log
#
cuda.tile.
log
(
x
,
/
)
#
Perform
log
on a tile.
Parameters
:
x
(
Tile
)
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
32
,
32
),
3.0
,
dtype
=
ct
.
float32
)
>>>
tx
=
ct
.
log
(
tx
)

## cuda.tile.tanh — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.tanh.html

cuda.tile.tanh
#
cuda.tile.
tanh
(
x
,
/
,
*
,
rounding_mode
=
None
)
#
Perform
tanh
on a tile.
Parameters
:
x
(
Tile
)
rounding_mode
(
RoundingMode
) –
Supported values:
RoundingMode.FULL
RoundingMode.APPROX
(since CTK 13.2)
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
32
,
32
),
3.0
,
dtype
=
ct
.
float32
)
>>>
tx
=
ct
.
tanh
(
tx
)
>>>
tx
=
ct
.
tanh
(
tx
,
rounding_mode
=
RoundingMode
.
APPROX
)
# Faster approximation

## cuda.tile.rsqrt — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.rsqrt.html

cuda.tile.rsqrt
#
cuda.tile.
rsqrt
(
x
,
/
,
*
,
flush_to_zero
=
False
)
#
Perform
rsqrt
on a tile.
Parameters
:
x
(
Tile
)
flush_to_zero
(
const bool
) – If True, flushes subnormal inputs and results to sign-preserving zero, default is False.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
32
,
32
),
3.0
,
dtype
=
ct
.
float32
)
>>>
tx
=
ct
.
rsqrt
(
tx
)

## cuda.tile.expand_dims — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.expand_dims.html

cuda.tile.expand_dims
#
cuda.tile.
expand_dims
(
x
,
/
,
axis
)
#
Reshapes the tile by inserting a new axis of size 1 at given position.
This can also be done via the NumPy-style syntax:
x[:, None]
or
x[np.newaxis, :]
Parameters
:
x
(
Tile
) – input tile.
axis
(
const int
) – axis to expand the tile dimension.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
arange
(
16
,
dtype
=
ct
.
float32
)
>>>
tx
.
shape
(16,)
>>>
ty
=
ct
.
expand_dims
(
x
,
1
)
>>>
ty
.
shape
(16,1)
>>>
ty
=
x
[
None
,
...
,
None
,
None
]
>>>
ty
.
shape
(1, 16, 1, 1)

## cuda.tile.reshape — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.reshape.html

cuda.tile.reshape
#
cuda.tile.
reshape
(
x
,
/
,
shape
)
#
Reshapes a tile to the specified shape.
One of the shape elements may be specified as -1 to indicate that the
corresponding dimension is to be inferred automatically.
For example, reshaping a
(16,
2)
tile to
(8,
-1)
will
produce a tile of shape
(8,
4)
: as there are 32 elements in total,
the second dimension will be computed as 32 divided by 8.
Parameters
:
x
(
Tile
) – input tile.
shape
(
tuple
[
const int
,
...
]
) – target shape.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
arange
(
8
,
dtype
=
ct
.
float32
)
>>>
tx
.
shape
(8,)
>>>
ty
=
ct
.
reshape
(
tx
,
(
2
,
4
))
>>>
ty
.
shape
(2, 4)
>>>
tz
=
ct
.
reshape
(
tx
,
(
2
,
-
1
))
>>>
tz
.
shape
(2, 4)

## cuda.tile.broadcast_to — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.broadcast_to.html

cuda.tile.broadcast_to
#
cuda.tile.
broadcast_to
(
x
,
/
,
shape
)
#
Broadcasts a tile to the specified shape
following
Numpy broadcasting rule
.
Parameters
:
x
(
Tile
) – input tile.
shape
(
tuple
[
const int
,
...
]
) – target shape.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
arange
(
4
,
dtype
=
ct
.
float32
)
>>>
tx
.
shape
(4,)
>>>
ty
=
ct
.
broadcast_to
(
tx
,
(
2
,
4
))
>>>
ty
.
shape
(2, 4)

## cuda.tile.maximum — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.maximum.html

cuda.tile.maximum
#
cuda.tile.
maximum
(
x
,
y
,
/
,
*
,
flush_to_zero
=
False
)
#
Elementwise maximum on two tiles.
Can also use builtin operation
max(x, y)
.
Parameters
:
x
(
Tile
) – LHS tile.
y
(
Tile
) – RHS tile.
flush_to_zero
(
const bool
) – If True, flushes subnormal inputs and results to sign-preserving zero, default is False.
The
shape
of
x
and
y
will be broadcasted and
dtype
promoted to common dtype.
Return type
:
Tile
Examples
>>>
# tile and tile
>>>
tx
=
ct
.
full
((
2
,
4
),
7
,
dtype
=
ct
.
int32
)
>>>
ty
=
ct
.
full
((
2
,
4
),
3
,
dtype
=
ct
.
int32
)
>>>
tz
=
ct
.
maximum
(
tx
,
ty
)
>>>
# Can also use the builtin op
>>>
tz
=
max
(
tx
,
ty
)
>>>
# shape broadcast
>>>
tx
=
ct
.
full
((
2
,
4
),
7
,
dtype
=
ct
.
int32
)
>>>
ty
=
ct
.
full
((
2
,),
3
,
dtype
=
ct
.
int32
)
>>>
tz
=
max
(
tx
,
ty
)
>>>
# dtype cast
>>>
tx
=
ct
.
full
((
2
,
4
),
7
,
dtype
=
ct
.
int32
)
>>>
ty
=
ct
.
full
((
2
,
4
),
3
,
dtype
=
ct
.
int64
)
>>>
tz
=
max
(
tx
,
ty
)
>>>
# tile and scalar
>>>
tx
=
ct
.
full
((
2
,
4
),
7
,
dtype
=
ct
.
int32
)
>>>
y
=
2
>>>
tz
=
max
(
tx
,
y
)
>>>
# scalar and scalar
>>>
z
=
max
(
7
,
2
)

## cuda.tile.truediv — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.truediv.html

cuda.tile.truediv
#
cuda.tile.
truediv
(
x
,
y
,
/
,
*
,
rounding_mode
=
None
,
flush_to_zero
=
False
)
#
Elementwise truediv on two tiles.
Can also use builtin operation
x / y
.
Parameters
:
x
(
Tile
) – LHS tile.
y
(
Tile
) – RHS tile.
rounding_mode
(
RoundingMode
) – The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
flush_to_zero
(
const bool
) – If True, flushes subnormal inputs and results to sign-preserving zero, default is False.
The
shape
of
x
and
y
will be broadcasted and
dtype
promoted to common dtype.
Return type
:
Tile
Examples
>>>
# tile and tile
>>>
tx
=
ct
.
full
((
2
,
4
),
7
,
dtype
=
ct
.
int32
)
>>>
ty
=
ct
.
full
((
2
,
4
),
3
,
dtype
=
ct
.
int32
)
>>>
tz
=
ct
.
truediv
(
tx
,
ty
)
>>>
# Can also use the builtin op
>>>
tz
=
tx
/
ty
>>>
# shape broadcast
>>>
tx
=
ct
.
full
((
2
,
4
),
7
,
dtype
=
ct
.
int32
)
>>>
ty
=
ct
.
full
((
2
,),
3
,
dtype
=
ct
.
int32
)
>>>
tz
=
tx
/
ty
>>>
# dtype cast
>>>
tx
=
ct
.
full
((
2
,
4
),
7
,
dtype
=
ct
.
int32
)
>>>
ty
=
ct
.
full
((
2
,
4
),
3
,
dtype
=
ct
.
int64
)
>>>
tz
=
tx
/
ty
>>>
# tile and scalar
>>>
tx
=
ct
.
full
((
2
,
4
),
7
,
dtype
=
ct
.
int32
)
>>>
y
=
2
>>>
tz
=
tx
/
y
>>>
# scalar and scalar
>>>
z
=
7
/
2

## cuda.tile.permute — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.permute.html

cuda.tile.permute
#
cuda.tile.
permute
(
x
,
/
,
axes
)
#
Permutes the axes of the input tile.
Parameters
:
x
(
Tile
) – input tile.
axes
(
tuple
[
const int
,
...
]
) – the desired axes order.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
2
,
4
,
8
),
0.
,
dtype
=
ct
.
float32
)
>>>
ty
=
ct
.
permute
(
tx
,
(
0
,
2
,
1
))
>>>
ty
.
shape
(2, 8, 4)

## cuda.tile.cat — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.cat.html

cuda.tile.cat
#
cuda.tile.
cat
(
tiles
,
/
,
axis
)
#
Concatenates two tiles along the
axis
.
Parameters
:
tiles
(
tuple
) – a pair of tiles to concatenate.
axis
(
const int
) – axis to concatenate the tiles.
Return type
:
Tile
Notes
Due to power-of-two assumption on all tile shapes,
the two input tiles must have the same shape.
Examples
>>>
tx
=
ct
.
full
((
2
,
4
),
3.
,
dtype
=
ct
.
float32
)
>>>
ty
=
ct
.
full
((
2
,
4
),
4.
,
dtype
=
ct
.
float32
)
>>>
tz
=
ct
.
cat
((
tx
,
ty
),
0
)
>>>
tz
.
shape
(4,4)
>>>
tz
=
ct
.
cat
((
tx
,
ty
),
1
)
>>>
tz
.
shape
(2,8)

## cuda.tile.scatter — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.scatter.html

cuda.tile.scatter
#
cuda.tile.
scatter
(
array
,
indices
,
value
,
/
,
*
,
mask
=
None
,
check_bounds
=
True
,
latency
=
None
,
)
#
Stores a tile
value
into the
array
elements specified by
indices
.
indices
must be a tuple whose length equals the
array
rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
value
must be a scalar or a tile whose shape is broadcastable to the
common shape of
indices
.
For example, consider a 2-dimensional array. In this case, indices must be a tuple
of length 2. Suppose that
ind0
and
ind1
are integer tiles
of shapes
(M,
N,
1)
and
(M,
1,
K)
, and
value
is a tile of shape of
(N,
K)
:
>>>
# ind0: (M, N, 1),  ind1: (M, 1, K),  value: (N, K)
>>>
ct
.
scatter
(
array
,
(
ind0
,
ind1
),
value
)
The above call to
scatter
will store elements according to
array
[
ind0
[
i
,
j
,
0
],
ind1
[
i
,
0
,
k
]]
=
value
[
j
,
k
]
If the array is 1-dimensional,
indices
can be passed as a tile rather than a tuple.
This is a convenience notation that is strictly equivalent to passing a tuple of length 1:
>>>
ct
.
scatter
(
array
,
ind0
,
value
)
# equivalent to ct.scatter(array, (ind0,), value)
A custom boolean
mask
can be provided to control which elements are stored.
The mask must be a scalar or a tile whose shape is broadcastable to the common shape
of indices. Where the mask is
False
, no store occurs.
scatter()
checks that indices are within the bounds of the array. For indices
that are out of bounds, nothing is stored.
If both
mask
and
check_bounds=True
are provided, the effective mask is the logical
AND of both the custom mask and the bounds-checking mask. This means an element is only
stored if both the custom mask is
True
AND the index is within bounds.
To disable bounds checking, set
check_bounds
to
False
. In this mode, the caller
is responsible for ensuring that all indices are within the bounds of the array, and
any out-of-bounds access will result in undefined behavior.

## cuda.tile.prod — cuTile Python

Source URL: https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.prod.html

cuda.tile.prod
#
cuda.tile.
prod
(
x
,
/
,
axis
=
None
,
*
,
keepdims
=
False
,
rounding_mode
=
None
,
flush_to_zero
=
False
,
)
#
Performs prod reduction on tile along the
axis
.
Parameters
:
x
(
Tile
) – input tile.
axis
(
None
|
const int
|
tuple
[
const int
,
...
]
) – the axis for reduction.
The default,
axis=None
, will reduce all of the elements.
keep_dims
(
const bool
) – If true, preserves the number of dimension
from the input tile.
rounding_mode
(
RoundingMode
) – The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
flush_to_zero
(
const bool
) – If True, flushes subnormal inputs and results to sign-preserving zero, default is False.
Return type
:
Tile
Examples
>>>
tx
=
ct
.
full
((
2
,
4
),
3
,
dtype
=
ct
.
float32
)
>>>
ty
=
ct
.
prod
(
tx
,
1
)
>>>
ty
.
shape
(2,)
>>>
ty
=
ct
.
prod
(
tx
,
1
,
keepdims
=
True
)
>>>
ty
.
shape
(2, 1)



---

# Evolved Skills (learned from TileGym)


## cutile-attention-kernels
_cuTile patterns specific to fused multi-head attention, flash decode, and attention sink kernels_

## cuTile Attention Kernel Patterns

- **FMHA tiling**: Factor attention into a reusable `_impl` device function that accepts block indices explicitly, enabling the same logic for both standard and attention-sink variants.
- **Online softmax**: Initialize `m_i = ct.full((TILE_M, 1), float('-inf'), ...)` and `l_i = ct.full((TILE_M, 1), 0.0, ...)` accumulators; update per K-tile with running max and sum correction.
- **Causal masking**: Use `ct.Constant[bool]` for `CAUSAL` flag to compile-time branch; generate causal masks from tile-local `ct.arange` offsets.
- **Grouped Query Attention (GQA)**: Multiple Q heads share one KV head via `NUM_Q_HEAD_PER_KV`. Load Q as `(1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM)`, then `ct.reshape()` to 2D before MMA.
- **Split-K parallelism** (flash decode): Split the KV sequence across grid dimension, each CTA computes partial softmax; reduce across splits afterward.
- **Attention sinks**: Load sink count as scalar via `ct.load(Sinks, ...).item()`; handle initial sink tokens separately before processing the main KV range.
- **TMA usage**: Always pass `allow_tma=True` and explicit `order` for Q/K/V loads to enable Tensor Memory Accelerator for efficient global→shared transfers.
- **Occupancy hint**: `@ct.kernel(occupancy=2)` for 2 concurrent CTAs per SM is typical for attention kernels.


## cutile-elementwise-and-1d-kernels
_cuTile patterns for elementwise and 1D operations: gather/scatter instead of load/store, explicit math functions with parameters, and proper type casting_

## cuTile Elementwise & 1D Kernel Patterns

- **Element access**: `ct.gather(tensor, (row_idx, col_offsets), check_bounds=True)` and `ct.scatter(tensor, (row_idx, col_offsets), value, check_bounds=True)`
- **Explicit math ops**: `ct.add(a, b)`, `ct.mul(a, b)`, `ct.truediv(a, b, flush_to_zero=True, rounding_mode='rn')` — no implicit operator overloading assumed
- **Type casting**: `ct.astype(tensor, target_dtype)` for explicit conversions
- **No built-in `ct.exp`/`ct.rand`**: Implement math functions manually or use available primitives
- **Bitwise ops**: `ct.bitwise_xor(a, b)` available for manual PRNG or hash functions
- **Index pattern**: `offsets = ct.arange(TILE_SIZE, dtype=ct.int32)` then `ct.gather(X, (block_idx, offsets))`
- **Bounds checking**: Always use `check_bounds=True` or `padding_value` instead of manual mask computation
- **Constants**: Annotate compile-time params as `ct.Constant[int]` or `ct.Constant[bool]` in kernel signatures


## cutile-kernel-patterns
_cuTile kernel design patterns: tile-based indexing, 2D tile shapes, MMA operations, and multi-dimensional grid strategies_

## cuTile Kernel Design Patterns

- **Tile-based indexing**: Use tile indices with `ct.load(tensor, (tile_m_idx, tile_k_idx), shape=(TILE_M, TILE_K))` — NOT pointer arithmetic like `A + offs_m[:, None] * K`
- **Tile counts**: `ct.num_tiles(tensor, dim, tile_shape)` to compute iteration bounds
- **Always use 2D tile shapes**: Even for vector operations, maintain proper 2D shapes `(TILE_M, TILE_K)` instead of 1D `[D]` with `None` broadcasting
- **Matrix multiply**: Use `ct.mma(a_tile, b_tile, accumulator)` for matrix multiply-accumulate
- **3D grid for attention**: Use `ct.bid(0)`, `ct.bid(1)`, `ct.bid(2)` for (batch, head, tile) instead of flattening into single `ct.program_id(0)` and manually computing indices
- **Masked/irregular access**: Use `ct.gather` with `padding_value` for out-of-bounds safety (e.g., LSE values in split attention)
- **TMA loads**: Add `order` and `allow_tma=True` params for efficient structured tensor memory access
- **Loop over K-tiles**: `for kk in range(num_k_tiles): tile = ct.load(A, (m_idx, kk), shape=...)`
