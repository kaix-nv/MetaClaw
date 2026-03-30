<!-- Auto-generated from cutile-python v1.2.0 by maintain.py. Do not edit manually. -->

## Load/Store

|  |  |
| --- | --- |
| bid | Gets the index of current block. |
| num\_blocks | Gets the number of blocks along the axis. |
| num\_tiles | Gets the number of tiles in the tile space of the array along the axis. Signature: `ct.num_tiles(array, axis, shape)` where `shape` is the **full tile shape** matching the array's rank. Example: for a 2D array A, use `ct.num_tiles(A, axis=1, shape=(BLOCK_M, BLOCK_K))`. Wrong: `ct.num_tiles(A.shape[1], BLOCK_K)` (old 2-arg scalar form does not exist). |
| load | Loads a tile from the array which is partitioned into a tile space. |
| store | Stores a tile value into the array at the index of its tile space. |
| gather | Loads a tile from the array elements specified by indices. |
| scatter | Stores a tile value into the array elements specified by indices. |
