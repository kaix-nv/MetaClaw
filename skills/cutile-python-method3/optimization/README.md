# cuTile Performance Optimization Guide

> **Acknowledgment**: This performance optimization guide was originally authored by **Yifei Song** and **Zhengyi Zhang**. Their production kernel development experience and insights form the foundation of these optimization patterns.

## Overview

This directory contains comprehensive performance optimization patterns discovered through production cuTile kernel development. **Read this before performance tuning.**

## Files

| File | Section | Description |
|------|---------|-------------|
| [p1_static_persistent_scheduling.md](p1_static_persistent_scheduling.md) | 1 | **HIGHEST IMPACT** - Static persistent scheduling patterns for GPU utilization |
| [p2_occupancy_autotune.md](p2_occupancy_autotune.md) | 2 | Occupancy hints and autotuning templates |
| [p3_tma_gather_scatter.md](p3_tma_gather_scatter.md) | 3 | TMA vs gather/scatter selection for memory access |
| [p4_dtype_and_indices.md](p4_dtype_and_indices.md) | 4-5 | DType consistency and gather/scatter index calculation |
| [p5_debugging_checklist.md](p5_debugging_checklist.md) | 6 | Performance debugging checklist |
| [p6_case_studies.md](p6_case_studies.md) | 7 | Real-world case studies (Softmax, Ragged BMM, Group GEMM) |
| [p7_anti_patterns.md](p7_anti_patterns.md) | 8 | Performance anti-patterns to avoid |
| [p8_optimization_priority.md](p8_optimization_priority.md) | 9-10 | Optimization priority checklist and quick reference |

## Quick Start

1. **Start with Section 1** - Static persistent scheduling gives the highest impact (2-4x speedup)
2. **Use Section 6** - Debugging checklist when kernel is slower than expected
3. **Consult Section 3** - For memory access pattern decisions (TMA vs gather/scatter)
4. **Reference Section 7** - For case studies of common optimization scenarios

## Priority Order

When optimizing a slow cuTile kernel:

1. **Algorithmic Issues** (10-100x impact) - Persistent scheduling, grid size
2. **Memory Access** (2-10x impact) - TMA vs gather/scatter selection
3. **Occupancy** (1.2-2x impact) - Kernel/launch occupancy match
4. **Microoptimizations** (1.05-1.2x impact) - Type conversions, loop hoisting
