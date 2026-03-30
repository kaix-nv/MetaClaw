# Section 9-10: Optimization Priority Checklist and Quick Reference

> Part of the cuTile Performance Optimization Guide. Originally authored by **Yifei Song** and **Zhengyi Zhang**.

---

## Optimization Priority Checklist

When optimizing a slow cuTile kernel, fix in this order:

### Priority 1: Algorithmic Issues (Can be 10-100x impact)
- [ ] Is persistent scheduling used?
- [ ] Is grid size reasonable (NUM_SM * occupancy)?
- [ ] Is work distribution balanced?

### Priority 2: Memory Access (Can be 2-10x impact)
- [ ] TMA vs gather/scatter: correct choice for access pattern?
- [ ] Are indices correctly calculated (Python ops, not ct ops)?
- [ ] Is coalescing optimal?

### Priority 3: Occupancy (Can be 1.2-2x impact)
- [ ] Does kernel decorator match launch occupancy?
- [ ] Is occupancy appropriate for workload type?

### Priority 4: Microoptimizations (Can be 1.05-1.2x impact)
- [ ] Minimize type conversions
- [ ] Hoist invariants out of loops
- [ ] Avoid redundant tensor creations

### Priority 5: Last Resort (Diminishing returns)
- [ ] Experiment with different BLOCK sizes
- [ ] Try different pipeline depths
- [ ] Profile with nsight compute

---

## Quick Reference

### Add Persistent Scheduling (30 seconds):
```python
# In kernel: change from bid to loop
- bid = ct.bid(0)
+ pid = ct.bid(0)
+ num_programs = ct.num_blocks(0)
+ for work_id in range(pid, total_work, num_programs):

# In launch: change grid
- grid = (n_items, 1, 1)
+ NUM_SM = torch.cuda.get_device_properties(device).multi_processor_count
+ grid = (NUM_SM * 4, 1, 1)

# In kernel signature: add total_work
- def kernel(input, output, ...):
+ def kernel(input, output, total_work: ct.Constant[int], ...):
```

### Choose TMA vs Gather (10 seconds decision):
```python
if block_aligned_access:
    use_tma()  # Faster
else:
    use_gather_scatter()  # Flexible
```

### Fix Slow Kernel (2 minutes):
1. Add `@ct.kernel(occupancy=4)`
2. Add persistent loop
3. Update grid to `(NUM_SM * 4, 1, 1)`
4. Test → Usually 2-3x faster
