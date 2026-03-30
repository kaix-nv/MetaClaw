# Section 1: Static Persistent Scheduling (HIGHEST IMPACT)

> Part of the cuTile Performance Optimization Guide. Originally authored by **Yifei Song** and **Zhengyi Zhang**.

---

**PROBLEM**: Naive 1:1 block-to-work mapping severely underutilizes GPU.

**BAD PATTERN** (Poor GPU Utilization):
```python
@ct.kernel
def naive_kernel(input, output, ...):
    bid = ct.bid(0)  # Each block processes ONE work item

    # Process single item
    data = ct.load(input, index=(bid, 0), ...)
    result = compute(data)
    ct.store(output, index=(bid, 0), tile=result)

# Launch: grid = (n_items, 1, 1)
# Problem: If n_items >> NUM_SM, thousands of blocks sit idle in queue
```

**GOOD PATTERN** (Static Persistent Scheduling):
```python
@ct.kernel(occupancy=4)  # ← Hint for resource allocation
def optimized_kernel(input, output, n_items: ct.Constant[int], ...):
    bid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    # Each block processes MULTIPLE items
    for item_idx in range(bid, n_items, num_programs):
        data = ct.load(input, index=(item_idx, 0), ...)
        result = compute(data)
        ct.store(output, index=(item_idx, 0), tile=result)

# Launch: grid = (NUM_SM * occupancy, 1, 1)
# Benefit: Fixed number of blocks, each processes ~(n_items / grid_size) items
```

**GRID SIZE CALCULATION**:
```python
NUM_SM = torch.cuda.get_device_properties(device).multi_processor_count
occupancy = 4  # Should match @ct.kernel(occupancy=X)
num_programs = min(NUM_SM * occupancy, total_work_items)
grid = (num_programs, 1, 1)
```

**EXPECTED PERFORMANCE GAIN**:
- Softmax: **+50-300%** (2-4x faster)
- Workloads with n_items > 1000: Typically **+100-200%**
- Best for row-wise/independent operations

**WHEN TO USE**:
- ✅ Row-wise operations (softmax, layer_norm, etc.)
- ✅ Independent work items (matmul tiles, attention blocks)
- ✅ When work_items >> NUM_SM
- ❌ When work_items < NUM_SM (just use grid=(work_items,))
