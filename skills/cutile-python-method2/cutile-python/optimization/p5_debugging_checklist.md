# Section 6: Performance Debugging Checklist

> Part of the cuTile Performance Optimization Guide. Originally authored by **Yifei Song** and **Zhengyi Zhang**.

---

When cuTile kernel is significantly slower than PyTorch:

## Step 1: Check Persistent Scheduling
```bash
# Look for this pattern:
grep "for.*in range(pid" your_kernel.py
```
✗ If missing → Add persistent loop (see p1_static_persistent_scheduling.md)

## Step 2: Check Grid Size
```bash
# Look for:
grep "grid = (n_items" your_launch.py
```
✗ If `grid = (n_items, 1, 1)` → Use occupancy-aware grid (see p1_static_persistent_scheduling.md)

## Step 3: Check Occupancy Match
```python
# Kernel decorator
@ct.kernel(occupancy=X)

# Launch function
occupancy = Y  # ← Must match X!
```
✗ If mismatch → Fix to match

## Step 4: Check Memory Access Pattern
- Are indices block-aligned? → Use TMA
- Are indices arbitrary offsets? → Use gather/scatter

See p3_tma_gather_scatter.md for decision tree.

## Step 5: Profile
```python
# Add to launch:
torch.cuda.synchronize()
start = time.time()
ct.launch(...)
torch.cuda.synchronize()
elapsed = time.time() - start
```
