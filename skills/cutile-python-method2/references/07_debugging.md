<!-- Auto-generated from cutile-python v1.2.0 by maintain.py. Do not edit manually. -->

# Debugging

## Exception Types

*class* cuda.tile.TileSyntaxError
:   Exception when a python syntax not supported by cuTile is encountered.

*class* cuda.tile.TileTypeError
:   Exception when an unexpected type or data type is encountered.

*class* cuda.tile.TileValueError
:   Exception when an unexpected python value is encountered.

*class* cuda.tile.TileUnsupportedFeatureError
:   Exception when a feature is not supported by the underlying compiler or
    the GPU architecture.

*class* cuda.tile.TileCompilerExecutionError
:   Exception when `tileiras` compiler throws an error.

*class* cuda.tile.TileCompilerTimeoutError
:   Exception when `tileiras` compiler timeout limit is exceeded.

## Environment Variables

The following environment variables are useful when
the above exceptions are encountered during kernel
development.

Set `CUDA_TILE_ENABLE_CRASH_DUMP=1` to enable dumping
an archive including the TileIR bytecode
for submitting a bug report on TileCompilerExecutionError
or TileCompilerTimeoutError.

Set `CUDA_TILE_COMPILER_TIMEOUT_SEC` to limit the
time the TileIR compiler tileiras can take.

Set `CUDA_TILE_LOGS=CUTILEIR` to print cuTile Python
IR during compilation to stderr. This is useful when
debugging TileTypeError.

Set `CUDA_TILE_TEMP_DIR` to configure the directory
for storing temporary files.

Set `CUDA_TILE_CACHE_DIR` to configure the directory
for the bytecode-to-cubin disk cache. Compiled cubins
are cached here to avoid recompilation of unchanged
kernels. Set to `0`, `off`, `none`, or an empty
string to disable caching. Defaults to
`~/.cache/cutile-python`.

Set `CUDA_TILE_CACHE_SIZE` to configure the maximum
disk cache size in bytes. Oldest entries are evicted
when the cache exceeds this limit. Defaults to
2 GB (2147483648).
