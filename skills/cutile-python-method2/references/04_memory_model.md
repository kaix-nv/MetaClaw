<!-- Auto-generated from cutile-python v1.2.0 by maintain.py. Do not edit manually. -->

# Memory Model

cuTile employs a memory model that permits the compiler and hardware to reorder operations for performance.
As a result, without explicit synchronization, there is no guaranteed ordering of memory accesses across threads.

To coordinate memory accesses among threads, cuTile provides two attributes for atomic operations:

- **Memory Order**: Defines the memory ordering semantics of an atomic operation.
- **Memory Scope**: Defines the scope of threads that participate in memory ordering.

Synchronization occurs at a per-element granularity. Each element in the array participates independently in the memory model.

For a more detailed explanation, see the Memory Model section in the Tile IR documentation.

## Memory Order

*class* cuda.tile.MemoryOrder
:   Memory ordering semantics of an atomic operation.

    RELAXED *= 'relaxed'*
    :   No ordering guarantees. Cannot be used to synchronize between threads.

    ACQUIRE *= 'acquire'*
    :   Acquire semantics. When this reads a value written by a release,
        the releasing thread’s prior writes become visible.
        Subsequent reads/writes within the same block cannot be reordered before this operation.

    RELEASE *= 'release'*
    :   Release semantics. When an acquire reads the value written by this,
        this thread’s prior writes become visible to the acquiring thread.
        Prior reads/writes within the same block cannot be reordered after this operation.

    ACQ\_REL *= 'acq\_rel'*
    :   Combined acquire and release semantics.

## Memory Scope

*class* cuda.tile.MemoryScope
:   The scope of threads that participate in memory ordering.

    BLOCK *= 'block'*
    :   Ordering guarantees apply to threads within the same block.

    DEVICE *= 'device'*
    :   Ordering guarantees apply to all threads on the same GPU.

    SYS *= 'sys'*
    :   Ordering guarantees apply to all threads across the entire system,
        including multiple GPUs and the host.
