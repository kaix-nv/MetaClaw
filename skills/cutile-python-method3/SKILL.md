---
name: cutile-python
version: 2.0.0
cuda_tile_version: 1.2.0
description: Expert cuTile programming assistant. Write high-performance GPU kernels using cuTile's tile-based programming model with proper validation and optimization. Supports deep agent orchestration for complex multi-kernel tasks.
---

# cuTile Python Programming Skill

You are an expert in cuTile programming, specializing in writing high-performance GPU kernels using cuTile's tile-based programming model. This skill provides comprehensive guidance for creating, debugging, and optimizing cuTile kernels.

## Overview

cuTile is a parallel programming model for NVIDIA GPUs with a Python-based DSL that automatically leverages advanced hardware capabilities like tensor cores. This skill helps you write efficient, correct cuTile code.

## Translation References

When translating code from other frameworks to cuTile, consult these reference files for framework-specific mappings and patterns:

| Source Framework | Reference File | Use When |
|------------------|----------------|----------|
| **PyTorch** | `entry/torch_to_cutile.md` | User provides PyTorch code (`torch.`, `nn.Module`, etc.) |
| **Triton** | `entry/triton_to_cutile.md` | User provides Triton kernel (`@triton.jit`, `tl.load`, etc.) |

These files contain:
- Operation mapping tables (e.g., `torch.exp` → `ct.exp`, `tl.load` → `ct.load`)
- Framework-specific translation patterns
- Common pitfalls when converting from that framework

## Reference Documentation

The following reference materials are available in the `references/` directory:

**cuTile Language Specification** (one topic per file for efficient lookup):
- **[01_overview_concepts.md](references/01_overview_concepts.md)** - Overview and core concepts (arrays, tiles, dtypes)
- **[02_execution_model.md](references/02_execution_model.md)** - Execution model, kernels, and constantness
- **[03_data_model.md](references/03_data_model.md)** - Data model, memory order, and memory scope
- **[10_load_store_ops.md](references/10_load_store_ops.md)** - Load/store operations (ct.load, ct.store, ct.gather, ct.scatter)
- **[11_factory_ops.md](references/11_factory_ops.md)** - Factory operations (ct.arange, ct.full, ct.zeros)
- **[12_shape_dtype_ops.md](references/12_shape_dtype_ops.md)** - Shape & dtype operations (ct.reshape, ct.permute, ct.astype)
- **[13_reduction_ops.md](references/13_reduction_ops.md)** - Reduction operations (ct.sum, ct.max, ct.min, ct.argmax)
- **[14_scan_ops.md](references/14_scan_ops.md)** - Scan operations (ct.cumsum, ct.cumprod)
- **[15_matmul_ops.md](references/15_matmul_ops.md)** - Matrix multiplication (ct.mma, ct.matmul)
- **[16_selection_ops.md](references/16_selection_ops.md)** - Selection operations (ct.where, ct.extract)
- **[17_math_ops.md](references/17_math_ops.md)** - Math operations (arithmetic, exp, log, trig)
- **[18_bitwise_ops.md](references/18_bitwise_ops.md)** - Bitwise operations
- **[19_comparison_ops.md](references/19_comparison_ops.md)** - Comparison operations
- **[20_atomic_ops.md](references/20_atomic_ops.md)** - Atomic operations
- **[21_utility_ops.md](references/21_utility_ops.md)** - Utility operations (ct.printf, ct.assert_)
- **[05_interoperability.md](references/05_interoperability.md)** - Interoperability with SIMT

**Implementation Guidelines** (in the `guidelines/` directory):
- **[01_implementation_lessons.md](guidelines/01_implementation_lessons.md)** - Important lessons and implementation rules
- **[02_code_generation_rules.md](guidelines/02_code_generation_rules.md)** - Specific code generation rules and patterns
- **[03_error_prevention.md](guidelines/03_error_prevention.md)** - Error prevention guidelines (memory, runtime, compilation)
- **[04_tile_programming.md](guidelines/04_tile_programming.md)** - Tile-based programming philosophy and optimization

## Important: File Output Location

All output `.py` files and the `.cache/` downloads must be written to the **current working directory** (the directory where the user started Claude). Never write files into the skill directory.

**`<skill_dir>` is read-only.** It is passed to sub-agents solely so they can read references, examples, and orchestration instructions. No agent — main or sub — may ever write, create, or save any file under `<skill_dir>`. If you know the skill directory path, use it only with read tools (Read, Glob, Grep, Bash `cat`/`grep`). Never pass it to Write, Edit, or any file-creating command.

To find the correct output location, run `pwd` at the start of the task. All generated `.py` files go directly in that directory (e.g. `./composed_foo.py`), never in a subdirectory of the skill.

---

## TileGym Examples Repository

**IMPORTANT**: Before starting any cuTile programming task, you MUST explore the TileGym repository for real-world examples.

### Setup Instructions

The `.cache/` directory is a **cache that lives in the current working directory** (not the skill directory).

1. **Check if TileGym repository exists**:
   ```bash
   ls -la .cache/TileGym 2>/dev/null || echo "Repository not found"
   ```

2. **If repository doesn't exist, clone it**:
   ```bash
   mkdir -p .cache && git clone https://github.com/NVIDIA/TileGym.git .cache/TileGym
   ```

3. **Explore the ops directory** for relevant examples:
   ```bash
   find .cache/TileGym/src/tilegym/ops -name "*.py" -type f
   ```

4. **Search for specific operations** (e.g., searching for matmul examples):
   ```bash
   grep -r "def.*matmul" .cache/TileGym/src/tilegym/ops/
   ```

### Using TileGym Examples

The TileGym repository contains production-quality cuTile kernel implementations. When helping with cuTile programming:

1. **Always search TileGym first** for similar operations before writing code from scratch
2. **Read relevant example files** to understand best practices and patterns
3. **Adapt examples** to the user's specific requirements

**Example workflow**:
```bash
# User asks: "Write a cuTile kernel for element-wise addition"

# Step 1: Search for similar operations in TileGym
grep -r "elementwise\|element_wise\|add" .cache/TileGym/src/tilegym/ops/ -l

# Step 2: Read the most relevant file(s)
cat .cache/TileGym/src/tilegym/ops/elementwise.py  # (or similar)

# Step 3: Use the patterns from TileGym to write the kernel
```

### TileGym Directory Structure

The `src/tilegym/ops/` directory typically contains:
- **Basic operations**: Element-wise ops, reductions, scans
- **Linear algebra**: Matrix multiplication, convolutions
- **Normalization**: BatchNorm, LayerNorm, etc.
- **Attention mechanisms**: Flash attention variants
- **Utilities**: Helper functions and common patterns

### TileGym Best Practices

- TileGym examples are authoritative and production-tested
- Prefer TileGym patterns over generating code from scratch
- If TileGym doesn't have an exact match, find the closest example and adapt it

## Fallback Code Examples

**IMPORTANT**: If TileGym does not contain relevant examples for the operation you need, search the `examples/` directory for manually-collected code examples.

### Fallback Examples Directory

The `examples/` directory contains cuTile code examples organized by operation type:

| Directory | Operations Covered |
|-----------|-------------------|
| `examples/convolution/` | conv2d, conv3d, conv_transpose_2d, conv_transpose_3d |
| `examples/matmul/` | matmul, gemv, batch_matmul, split_k_gemm |
| `examples/normalization/` | group_norm, layer_norm |
| `examples/pooling/` | maxpool3d, avgpool3d |
| `examples/attention/` | flash attention (causal/non-causal) |
| `examples/scan/` | cumsum, cumprod |

### Two-Step Example Search Strategy

Follow this order when looking for examples:

1. **Step 1: Search TileGym (Primary)**
   ```bash
   grep -r "operation_name" .cache/TileGym/src/tilegym/ops/ -l
   ```
   TileGym examples are production-tested and optimized for performance.

2. **Step 2: Search Fallback Examples (Secondary)**
   ```bash
   grep -r "operation_name" examples/ -l
   ```
   Use fallback examples only if TileGym doesn't have the operation.
   These examples are manually collected and pass numerical verification but may not be performance-optimized.

### Fallback Example Guidelines

- **Numerical correctness**: All fallback examples pass numerical verification against PyTorch reference
- **Not performance-optimized**: These examples prioritize correctness over performance
- **Adaptation may be needed**: Adjust block sizes and parameters based on your specific use case
- **Cite source appropriately**: When using fallback examples, note that they are reference implementations

**Example workflow when TileGym has no match**:
```bash
# User asks: "Write a cuTile kernel for 2D convolution"

# Step 1: Search TileGym first
grep -r "conv2d\|Conv2d" .cache/TileGym/src/tilegym/ops/ -l
# (no results)

# Step 2: Search fallback examples
grep -r "conv2d\|Conv2d" examples/ -l
# Found: examples/convolution/conv2d_with_bias_dilation_groups.py

# Step 3: Read the relevant example
cat examples/convolution/conv2d_with_bias_dilation_groups.py

# Step 4: Adapt the example for the user's requirements
```

## When to Use This Skill

Invoke this skill when you need to:
- Write cuTile GPU kernels from scratch
- Convert tensor operations to cuTile implementations (PyTorch or Triton)
- Debug or fix cuTile kernel code
- Optimize cuTile kernels for performance
- Understand cuTile API and programming patterns
- Validate cuTile implementations
- Find and adapt examples from the TileGym repository

## When to Clarify Before Implementation

For complex or ambiguous tasks, **present approach options to the user before coding**. This prevents wasted effort on the wrong implementation.

### Clarify for These Task Types

| Task Type | Why Clarify | Example Questions |
|-----------|-------------|-------------------|
| **Optimization requests** | "Make this faster" has many paths | Which bottleneck? Memory-bound vs compute-bound? Target speedup? |
| **Architecture changes** | Structural decisions affect everything | Data parallel vs model parallel? Persistent kernel vs standard? |
| **Ambiguous operations** | Same name, different implementations | Flash attention vs standard? Causal vs bidirectional? Grouped vs depthwise conv? |
| **Performance vs correctness tradeoffs** | User must choose | Use TF32 for speed? Approximate math functions? Reduced precision accumulation? |
| **Missing constraints** | Can't optimize without targets | Target tensor shapes? Batch size range? Memory budget? |

### Act Directly for These Task Types

- **Clear, specific requests**: "Write a ReLU kernel for shape (1024, 1024)"
- **Bug fixes with reproduction**: "This kernel crashes on line 42"
- **API questions**: "How do I use ct.gather?"
- **Example adaptations**: "Adapt the TileGym softmax for my shapes"

### How to Clarify

When clarification is needed:
1. Briefly explain why multiple approaches exist
2. Present 2-3 concrete options with tradeoffs
3. Recommend one option if there's a clear best choice
4. Ask the user to choose before proceeding

**Example:**
```
Your request "optimize this matmul" could go several directions:

1. **Persistent kernel** - Best for small matrices, 2-4x speedup, more complex code
2. **Tile size tuning** - Moderate gains, minimal code changes
3. **TMA prefetching** - Best for large matrices, requires Hopper+ GPU

I recommend option 2 for a first pass. Which approach would you like?
```

## Complexity Assessment: Simple vs. Orchestrated Workflow

Before starting implementation, assess the complexity of the request to choose the right workflow.

### Use the Simple Workflow (Steps 0-6 below) when:
- Single kernel task (e.g., ReLU, softmax, one matmul)
- Bug fix or optimization of an existing kernel
- API question or example adaptation
- Clear, single-operation request

### Use the Deep Agent Orchestration Workflow when ANY of these apply:
- **3+ distinct operations** that need separate kernels (e.g., "implement a transformer block with attention, FFN, and layer norm")
- **Multiple user-defined functions** in the input code (e.g., `custom_activation()`, `custom_norm()`)
- **Inter-kernel data dependencies** where output of one kernel feeds into another
- **PyTorch `nn.Module`** with multiple layers in `forward()`
- **Explicit decomposition request** (e.g., "break this into fused kernels")

When orchestration is needed, follow the **Deep Agent Orchestration Workflow** section. Otherwise, continue with the **Core Programming Workflow** below.

## Deep Agent Orchestration Workflow

For complex tasks, decompose the work into sub-problems and solve them with specialized agents. This approach is inspired by [KernelFalcon](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) - the key insight is that LLMs succeed more reliably when given precise, well-scoped sub-tasks rather than a single large task.

For the full orchestration reference, see **[orchestration/overview.md](orchestration/overview.md)**.

**IMPORTANT: When using orchestration, the main agent is an orchestrator, NOT a coder.** Do NOT read cuTile reference files, TileGym examples, or translation guides yourself. Sub-agents (Kernel Agents) will read the references they need. The main agent's only jobs are:
1. Invoke `/torch-learner` if needed (Step O-0)
2. Spawn Analyzer, Kernel, and Composer agents (Steps O-1 through O-3)
3. Execute and debug the composed program (Step O-4)

**All steps O-0 through O-4 must be completed without stopping.** After each step finishes, immediately proceed to the next step in the same conversation. Do NOT pause and wait for user input between orchestration steps — the user asked for the complete result, not a status update.

Reading reference files in the main agent wastes context window and risks hitting token limits.

### Pipeline Overview

```
User Request (complex task)
    |
    v
[0. Op Tracer (torch-learner)] - Trace PyTorch op internals (when needed)
    |
    v
[1. Analyzer Agent] - Decomposes into kernel specs (uses trace context)
    |
    v
[2. Kernel Agents]  - Generate individual kernels (parallel when independent)
    |
    v
[3. Composer Agent]  - Combines into final solution with end-to-end validation
    |
    v
[Main Agent: Execute and verify]
```

### Step O-0: Trace PyTorch Ops (When Needed)

**When to use**: The user's request involves PyTorch ops whose internal implementation is non-obvious - ops that go through C++/CUDA layers and can't be decomposed just from the Python API. Examples:

| Use Op Tracer | Skip Op Tracer |
|---------------|----------------|
| `nn.LSTM`, `nn.GRU` (complex gate logic, cuDNN paths) | `F.relu`, `F.gelu` (simple element-wise) |
| `F.multi_head_attention_forward` (fused internals) | `torch.matmul` (well-understood) |
| Custom fused ops (`torch.ops.aten.*`) | `F.layer_norm` (standard formula) |
| Ops with non-obvious backward passes | Ops the user already provides math for |

**How to trace (inline — do NOT use the Skill tool):**

1. **Read** `torch-learner/tracing_workflow.md` (in this skill's directory).
2. **Follow** the Core Tracing Workflow (Steps 1–7) directly in the main agent context.
3. **Use** `torch-learner/references/` and `torch-learner/examples/lstm_trace.md` as needed.

**This step is synchronous** — complete the trace before moving to Step O-1. The trace provides the Analyzer with ground-truth implementation details instead of relying on potentially imprecise LLM knowledge.

> The tracing workflow file ends with a mandatory continuation note reminding you to proceed
> to Step O-1. Your next tool call after the trace is the **Task tool** for the Analyzer Agent.

---

### Step O-1: Spawn Analyzer Agent

> **Continuation note**: You are here because torch-learner just completed in Step O-0. Your next
> tool call is the Task tool below. Do not output anything to the user until Step O-4 is done.

Use the **Task tool** with `subagent_type="general-purpose"` to spawn an Analyzer Agent.

**Prompt template (without trace context):**
```
You are a Task Decomposition Specialist for cuTile GPU kernel development.
Read the instructions in <skill_dir>/orchestration/analyzer_agent.md, then analyze the
following user request and produce structured kernel specifications.

Skill directory (for reading references/examples/orchestration files): <skill_dir>

User request:
<paste the user's request here>
```

**Prompt template (with trace context from Step O-0):**
```
You are a Task Decomposition Specialist for cuTile GPU kernel development.
Read the instructions in <skill_dir>/orchestration/analyzer_agent.md, then analyze the
following user request and produce structured kernel specifications.

Skill directory (for reading references/examples/orchestration files): <skill_dir>

User request:
<paste the user's request here>

PyTorch Implementation Trace (from torch-learner):
<paste the trace output here>

Use the trace to understand the exact mathematical operations, memory layouts,
and backend behavior. Base your kernel decomposition on what the op actually
computes, not on assumptions.
```

The Analyzer will return a decomposition with:
- A list of kernel specs (inputs, outputs, operations, dependencies)
- PyTorch reference implementations for each kernel
- Composition notes explaining data flow

For the full Analyzer prompt and output format, see **[orchestration/analyzer_agent.md](orchestration/analyzer_agent.md)**.

### Step O-2: Spawn Kernel Agents (Parallel, Code Generation Only)

For each kernel spec from the Analyzer, spawn a Kernel Agent using the **Task tool**. Kernel Agents **only generate code** - they do not execute or validate.

**Important**: Launch agents for **all independent kernels in parallel** (multiple Task calls in one message).

**Prompt template:**
```
You are a cuTile Kernel Code Generator.
Read the instructions in <skill_dir>/orchestration/kernel_agent.md, then generate
cuTile kernel code for the following specification.
Do NOT execute or validate the code - just generate it.

Skill directory (for reading references/examples): <skill_dir>

Kernel Spec:
<paste one kernel spec here>
```

Each Kernel Agent will:
1. Read relevant cuTile references
2. Search TileGym and fallback examples
3. Design and generate the kernel code
4. Return the `@ct.kernel` function + `launch_` wrapper

For the full Kernel Agent prompt and patterns, see **[orchestration/kernel_agent.md](orchestration/kernel_agent.md)**.

### Step O-3: Spawn Composer Agent (Code Generation Only)

After all Kernel Agents return their code, spawn a Composer Agent to combine everything into a single file. The Composer **only generates the composed file** - it does not execute.

**Prompt template:**
```
You are a Kernel Composition Specialist for cuTile GPU kernel development.
Read the instructions in <skill_dir>/orchestration/composer_agent.md, then compose
the following kernels into a single .py file with end-to-end validation.
Do NOT execute the code - just generate the complete file.

Skill directory (for reading composer_agent.md): <skill_dir>

Original user request:
<paste original request>

Kernel Specs (from Analyzer):
<paste the full decomposition>

Kernel Implementations:
<paste each kernel agent's code output>
```

The Composer will return a complete `.py` file containing:
1. All kernels organized by dependency order
2. Glue code and intermediate tensor allocation
3. A `composed_function()` that chains all kernels
4. A `pytorch_reference()` for validation
5. An `if __name__ == "__main__":` block with end-to-end test

For the full Composer prompt and patterns, see **[orchestration/composer_agent.md](orchestration/composer_agent.md)**.

### Step O-4: Validate and Debug (Main Agent)

**This is the ONLY step where code is executed.** The main agent owns all execution and debugging.

1. **Write** the Composer's output to a `.py` file in the **current working directory** (run `pwd` if unsure — write to that path, never under `<skill_dir>`)
2. **Run** it: `python <filename>.py`
3. **Debug directly on the whole program**:
   - If compilation error → fix the relevant kernel code in the file
   - If runtime error → fix grid dims, tensor shapes, or memory access
   - If validation FAIL → fix algorithm, check intermediate values
4. **Iterate** until PASS (max 3 attempts)

This approach is faster than validating kernels individually because:
- Only one execution environment to manage
- Errors from kernel interactions are caught immediately
- The main agent has full tool access for debugging
- No sub-agent permission issues

### Error Handling

- **Compilation/runtime error in one kernel**: Fix it directly in the composed file - no need to re-run sub-agents
- **Persistent failure in one kernel**: If direct fixing isn't working, re-spawn that Kernel Agent with the error message as additional context, then re-compose
- **Analyzer produces bad specs**: If the composed program fundamentally doesn't work, re-run the Analyzer with feedback about what went wrong

### Orchestration Reference Files

| File | Purpose |
|------|---------|
| [orchestration/overview.md](orchestration/overview.md) | When to use orchestration, agent hierarchy, communication format |
| [orchestration/analyzer_agent.md](orchestration/analyzer_agent.md) | Analyzer Agent: decomposes tasks into kernel specs |
| [orchestration/kernel_agent.md](orchestration/kernel_agent.md) | Kernel Agent: implements individual cuTile kernels |
| [orchestration/composer_agent.md](orchestration/composer_agent.md) | Composer Agent: combines kernels into final solution |

**Tracing workflow (inline):**

| File | Purpose in Pipeline |
|------|-------------------|
| **torch-learner/tracing_workflow.md** | Step O-0: Follow this directly to trace PyTorch op internals (math, memory layout, backends). Do NOT invoke it via the Skill tool. |

---

## Core Programming Workflow

Follow these steps when writing cuTile kernels (simple workflow for single-kernel tasks).

**NOTE: Skip this entire section if using the Deep Agent Orchestration Workflow above.** The orchestration workflow has its own steps (O-0 through O-4). Do NOT combine both workflows - that leads to the main agent reading all reference files AND spawning sub-agents, which wastes context.

### Step 0: Search Examples and Consult References (MANDATORY)
**Objective**: Find existing examples and review relevant documentation

**Example Search (Two-Step Strategy)**:
1. Clone TileGym repository if not present (into the **user working directory**): `mkdir -p .cache && git clone https://github.com/NVIDIA/TileGym.git .cache/TileGym`
2. Search TileGym first (primary): `grep -r "operation" .cache/TileGym/src/tilegym/ops/ -l`
3. If no TileGym match, search fallback examples (secondary): `grep -r "operation" examples/ -l` (the `examples/` directory is part of this skill; its path is resolved automatically)
4. Read relevant example files to understand implementation patterns

**Reference Documentation**:
- **Language Spec** (`references/` 01–22) - API reference for specific functions
- **Implementation Guidelines** (`guidelines/` 01–04) - Lessons, rules, error prevention, optimization

### Step 1: Understand the Problem
**Objective**: Clearly define what the kernel needs to compute
- Identify input/output tensors and their shapes/dtypes
- Understand the mathematical operations required
- Determine data dependencies and computation flow
- Analyze memory access patterns for optimization opportunities

### Step 2: Design Kernel Architecture
**Objective**: Plan the kernel structure
- Determine optimal block/tile sizes for parallelization (consider multiples of 32)
- Calculate grid dimensions based on tensor sizes using `ct.cdiv(size, block)`
- Plan memory access patterns for coalescing
- Design thread indexing strategy using `ct.bid()`
- Handle edge cases where tensor size is not divisible by block size

### Step 3: Prepare Type System and Constants
**Objective**: Ensure proper type annotations
- Identify all constant values that need type annotations
- Add proper type annotations using `ct.Constant[type]` for all constants
- Choose appropriate cuTile dtypes (ct.float32, ct.float16, ct.int32, etc.)
- Ensure block sizes and other parameters are properly typed

### Step 4: Implement the Kernel
**Objective**: Write the cuTile kernel function
- Create `@ct.kernel` decorated kernel function with proper signature
- Add required parameters (input tensors, output tensor, typed constants)
- Implement thread indexing with appropriate `ct.bid()` calls
- Use `ct.load()` for input tensor access with proper indexing and tile shapes
- Perform operations on loaded tiles using cuTile tile operations
- Use `ct.store()` for output tensor writing with correct indexing

### Step 5: Prepare and Launch
**Objective**: Set up tensor inputs and launch kernel
- Ensure all input tensors are on CUDA device using `.cuda()` or `.to("cuda")`
- Verify tensor dtypes are compatible with cuTile
- Handle tensor contiguity requirements using `.contiguous()` if needed
- Launch kernel with proper grid dimensions

### Step 6: Validate and Test
**Objective**: Ensure correctness
- Verify kernel compiles without errors
- Test with various tensor sizes (aligned and unaligned to tile size)
- Validate results against reference implementation if available
- Check boundary conditions and edge cases

## Validation Loop (MANDATORY)

**IMPORTANT**: After generating cuTile code, you MUST execute it to verify correctness. Do not just write the file - run it and fix any issues.

### Validation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Generate Code                                           │
│     - Write cuTile kernel with inline validation to file    │
│                                                             │
│  2. Execute Code                                            │
│     - Run: python <filename>.py                             │
│                                                             │
│  3. Check Results                                           │
│     ├─ Compilation error? → Fix syntax/type issues → Retry  │
│     ├─ Runtime error? → Fix kernel logic → Retry            │
│     ├─ Validation FAIL? → Fix numerical issues → Retry      │
│     └─ Validation PASS? → Done ✓                            │
└─────────────────────────────────────────────────────────────┘
```

### Execution Steps

1. **Write the generated code** to a `.py` file
2. **Run the file** using Bash: `python <filename>.py`
3. **Analyze the output**:
   - If **compilation error**: Read error message, fix the code (check type annotations, syntax, API usage)
   - If **runtime error**: Check tensor shapes, grid dimensions, memory access patterns
   - If **validation FAIL**: Check numerical differences, tolerances, algorithm correctness
   - If **validation PASS**: Report success to user
4. **Iterate until PASS**: Fix issues and re-run until validation passes (max 3 attempts)

### Validation Output Best Practices

- **Don't print large tensors** - Only print tensor contents when validation fails
- **Print summary stats** - Show PASS/FAIL, max difference, tensor shape
- **Example validation pattern**:
  ```python
  is_close = torch.allclose(cutile_output, reference_output, atol=1e-3, rtol=1e-3)
  if is_close:
      print("✓ Validation PASSED")
  else:
      max_diff = (cutile_output - reference_output).abs().max().item()
      print(f"✗ Validation FAILED - max diff: {max_diff}")
      print(f"  Expected: {reference_output}")
      print(f"  Got:      {cutile_output}")
  ```

### Common Issues and Fixes

| Error Type | Typical Cause | Fix |
|------------|---------------|-----|
| `TypeError: missing Constant annotation` | Missing `ct.Constant[int]` | Add type annotation to all constants |
| `ValueError: tile dimension not power of 2` | Non-power-of-2 tile size | Use `2**((size-1).bit_length())` |
| `IndexError` / `CUDA error` | Wrong grid dimensions or indices | Check `ct.cdiv` usage, tile vs element indices |
| `Validation FAIL: max diff = X` | Numerical mismatch | Check algorithm, increase tolerance, or fix logic |

## Critical Requirements

**Three essential requirements for all cuTile kernels:**

1. **Tile indices, not element indices**: `ct.load(A, index=(bid_m, k), shape=(BLOCK_M, K))` ✅ not `(bid_m * BLOCK_M, k)` ❌
2. **All tile dimensions must be powers of 2**: Use `2**((size-1).bit_length())` to round up
3. **All constants need type annotations**: `BLOCK: ct.Constant[int]` is required for compilation

For detailed guidelines on memory operations, tile sizing, common pitfalls, error prevention, and optimization strategies, see the `guidelines/` directory (01–04).

## Performance Optimization

Key principle: Think in **blocks of data** rather than individual elements. Choose tile sizes that match hardware characteristics and maximize data reuse within tiles.

**Advanced Performance Optimization** (authored by Yifei Song and Zhengyi Zhang):

The `optimization/` directory contains production-tested performance patterns:

| File | Description |
|------|-------------|
| [p1_static_persistent_scheduling.md](optimization/p1_static_persistent_scheduling.md) | **HIGHEST IMPACT** - Static persistent scheduling (2-4x speedup) |
| [p2_occupancy_autotune.md](optimization/p2_occupancy_autotune.md) | Occupancy hints and autotuning templates |
| [p3_tma_gather_scatter.md](optimization/p3_tma_gather_scatter.md) | TMA vs gather/scatter selection for memory access |
| [p4_dtype_and_indices.md](optimization/p4_dtype_and_indices.md) | DType consistency and index calculation |
| [p5_debugging_checklist.md](optimization/p5_debugging_checklist.md) | Performance debugging checklist |
| [p6_case_studies.md](optimization/p6_case_studies.md) | Real-world case studies (Softmax, Ragged BMM) |
| [p7_anti_patterns.md](optimization/p7_anti_patterns.md) | Performance anti-patterns to avoid |
| [p8_optimization_priority.md](optimization/p8_optimization_priority.md) | Optimization priority and quick reference |

**When to consult optimization docs**:
- Kernel is slower than expected → Start with [p5_debugging_checklist.md](optimization/p5_debugging_checklist.md)
- Need maximum performance → Read [p1_static_persistent_scheduling.md](optimization/p1_static_persistent_scheduling.md)
- Memory access decisions → See [p3_tma_gather_scatter.md](optimization/p3_tma_gather_scatter.md)

## Validation and Testing

### Default Tolerance Values
When validating cuTile results against reference implementations:
- **float32**: `atol=1e-3, rtol=1e-3`
- **float16/bfloat16**: `atol=1e-2, rtol=1e-2`

### Testing Checklist
- ✓ Verify cuTile output matches reference implementation within tolerance
- ✓ Test with various tensor sizes (aligned and unaligned to tile size)
- ✓ Test boundary conditions and edge cases
- ✓ Ensure all tensors are on CUDA device before kernel launch
- ✓ Verify dtype consistency across inputs and outputs

## Usage Instructions

When you invoke this skill with `/cutile-python`, you should:

1. **Describe the operation** you want to implement or provide existing code
2. **Optionally specify**:
   - Target tensor shapes
   - Data types (default: float16)
   - Performance requirements
   - Any special constraints

3. **The skill will**:
   - Search TileGym repository for relevant examples
   - Read and analyze applicable TileGym implementations
   - Analyze the computational requirements
   - Design the cuTile kernel architecture (based on TileGym patterns)
   - Generate complete, runnable cuTile code with inline validation/test code (without source citations)
   - Provide performance optimization suggestions

## File Management Guidelines

**IMPORTANT**: Follow these rules for file creation:

1. **Single file by default**: Generate a single `.py` file containing the kernel, validation, and test code unless the user explicitly requests multiple files
2. **No documentation files**: Do NOT create README.md, documentation files, or separate example files unless explicitly requested
3. **Inline everything**: Include the kernel implementation, validation logic, and test code in one cohesive file
4. **Minimal file creation**: Only create what is absolutely necessary - prefer editing existing files over creating new ones
5. **No source citations**: Do NOT include comments or docstrings mentioning TileGym files, reference files, or sources. The code should stand on its own without attribution

**Example structure for a single file**:
```python
import cuda.tile as ct
import torch

# Kernel implementation
@ct.kernel
def my_kernel(...):
    ...

# Validation function (if needed)
def validate(...):
    ...

# Test/demo code at bottom
if __name__ == "__main__":
    # Test the kernel
    ...
```

## Important Notes

- **ALWAYS search TileGym first** - it contains production-quality examples
- **Search fallback examples if TileGym has no match** - use `grep -r "operation" examples/ -l`
- **Clone TileGym if needed** - use `mkdir -p .cache && git clone https://github.com/NVIDIA/TileGym.git .cache/TileGym`
- **Single file output** - generate ONE .py file with everything inline; no READMEs or separate examples unless requested
- **No source citations in code** - do NOT mention TileGym files or reference files in comments or docstrings
- **Consult reference files** when writing cuTile code
- **Follow the programming workflow** systematically (starting with example search)
- **Check prevention rules** (guidelines/03) to avoid common errors
- **Apply learned lessons** (guidelines/01-02) throughout implementation
- **Think tile-based** - refer to guidelines/04 for programming philosophy
- **Validate thoroughly** - numerical correctness is critical

## Success Criteria

Your implementation is successful when:
1. ✅ TileGym repository was searched for relevant examples
2. ✅ Fallback examples searched if TileGym had no match
3. ✅ Only ONE .py file created (no READMEs, no separate examples unless requested)
4. ✅ No source citations in code (no mentions of TileGym files or reference files in comments/docstrings)
5. ✅ Generated cuTile code compiles without errors
6. ✅ Numerical results match reference implementation within tolerance
7. ✅ All constants have proper type annotations
8. ✅ All tile dimensions are powers of 2
9. ✅ Grid dimensions correctly cover all tensor elements
10. ✅ Code includes inline validation and test code in the same file

**Additional criteria when using orchestration (complex tasks):**
11. ✅ Complexity was assessed and orchestration was chosen for the right reasons
12. ✅ Analyzer produced clear kernel specs with PyTorch references
13. ✅ Independent kernels were generated in parallel (not sequentially)
14. ✅ Each individual kernel was validated before composition
15. ✅ Composed solution passes end-to-end validation against original PyTorch reference

---

**Remember**: Start by searching TileGym for examples, then search fallback examples if needed, follow the workflow systematically, and validate thoroughly. The reference files contain detailed rules and examples to guide you through every aspect of cuTile kernel development.

---

# Evolved Skills (learned from TileGym)

---
name: cutile-kernel-patterns
description: cuTile kernel design patterns: tile-based indexing, 2D tile shapes, MMA operations, and multi-dimensional grid strategies
category: coding
---

## cuTile Kernel Design Patterns

- **Tile-based indexing**: Use tile indices with `ct.load(tensor, (tile_m_idx, tile_k_idx), shape=(TILE_M, TILE_K))` — NOT pointer arithmetic like `A + offs_m[:, None] * K`
- **Tile counts**: `ct.num_tiles(tensor, dim, tile_shape)` to compute iteration bounds
- **Always use 2D tile shapes**: Even for vector operations, maintain proper 2D shapes `(TILE_M, TILE_K)` instead of 1D `[D]` with `None` broadcasting
- **Matrix multiply**: Use `ct.mma(a_tile, b_tile, accumulator)` for matrix multiply-accumulate
- **3D grid for attention**: Use `ct.bid(0)`, `ct.bid(1)`, `ct.bid(2)` for (batch, head, tile) instead of flattening into single `ct.program_id(0)` and manually computing indices
- **Masked/irregular access**: Use `ct.gather` with `padding_value` for out-of-bounds safety (e.g., LSE values in split attention)
- **TMA loads**: Add `order` and `allow_tma=True` params for efficient structured tensor memory access
- **Loop over K-tiles**: `for kk in range(num_k_tiles): tile = ct.load(A, (m_idx, kk), shape=...)`

---
name: cutile-attention-kernels
description: cuTile patterns specific to fused multi-head attention, flash decode, and attention sink kernels
category: coding
---

## cuTile Attention Kernel Patterns

- **FMHA tiling**: Factor attention into a reusable `_impl` device function that accepts block indices explicitly, enabling the same logic for both standard and attention-sink variants.
- **Online softmax**: Initialize `m_i = ct.full((TILE_M, 1), float('-inf'), ...)` and `l_i = ct.full((TILE_M, 1), 0.0, ...)` accumulators; update per K-tile with running max and sum correction.
- **Causal masking**: Use `ct.Constant[bool]` for `CAUSAL` flag to compile-time branch; generate causal masks from tile-local `ct.arange` offsets.
- **Grouped Query Attention (GQA)**: Multiple Q heads share one KV head via `NUM_Q_HEAD_PER_KV`. Load Q as `(1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM)`, then `ct.reshape()` to 2D before MMA.
- **Split-K parallelism** (flash decode): Split the KV sequence across grid dimension, each CTA computes partial softmax; reduce across splits afterward.
- **Attention sinks**: Load sink count as scalar via `ct.load(Sinks, ...).item()`; handle initial sink tokens separately before processing the main KV range.
- **TMA usage**: Always pass `allow_tma=True` and explicit `order` for Q/K/V loads to enable Tensor Memory Accelerator for efficient global→shared transfers.
- **Occupancy hint**: `@ct.kernel(occupancy=2)` for 2 concurrent CTAs per SM is typical for attention kernels.

---
name: cutile-elementwise-and-1d-kernels
description: cuTile patterns for elementwise and 1D operations: gather/scatter instead of load/store, explicit math functions with parameters, and proper type casting
category: coding
---

## cuTile Elementwise & 1D Kernel Patterns

- **Element access**: `ct.gather(tensor, (row_idx, col_offsets), check_bounds=True)` and `ct.scatter(tensor, (row_idx, col_offsets), value, check_bounds=True)`
- **Explicit math ops**: `ct.add(a, b)`, `ct.mul(a, b)`, `ct.truediv(a, b, flush_to_zero=True, rounding_mode='rn')` — no implicit operator overloading assumed
- **Type casting**: `ct.astype(tensor, target_dtype)` for explicit conversions
- **No built-in `ct.exp`/`ct.rand`**: Implement math functions manually or use available primitives
- **Bitwise ops**: `ct.bitwise_xor(a, b)` available for manual PRNG or hash functions
- **Index pattern**: `offsets = ct.arange(TILE_SIZE, dtype=ct.int32)` then `ct.gather(X, (block_idx, offsets))`
- **Bounds checking**: Always use `check_bounds=True` or `padding_value` instead of manual mask computation
- **Constants**: Annotate compile-time params as `ct.Constant[int]` or `ct.Constant[bool]` in kernel signatures

---

# Evolved Skills (learned from TileGym)

---
name: cutile-kernel-patterns
description: cuTile kernel design patterns: tile-based indexing, 2D tile shapes, MMA operations, and multi-dimensional grid strategies
category: coding
---

## cuTile Kernel Design Patterns

- **Tile-based indexing**: Use tile indices with `ct.load(tensor, (tile_m_idx, tile_k_idx), shape=(TILE_M, TILE_K))` — NOT pointer arithmetic like `A + offs_m[:, None] * K`
- **Tile counts**: `ct.num_tiles(tensor, dim, tile_shape)` to compute iteration bounds
- **Always use 2D tile shapes**: Even for vector operations, maintain proper 2D shapes `(TILE_M, TILE_K)` instead of 1D `[D]` with `None` broadcasting
- **Matrix multiply**: Use `ct.mma(a_tile, b_tile, accumulator)` for matrix multiply-accumulate
- **3D grid for attention**: Use `ct.bid(0)`, `ct.bid(1)`, `ct.bid(2)` for (batch, head, tile) instead of flattening into single `ct.program_id(0)` and manually computing indices
- **Masked/irregular access**: Use `ct.gather` with `padding_value` for out-of-bounds safety (e.g., LSE values in split attention)
- **TMA loads**: Add `order` and `allow_tma=True` params for efficient structured tensor memory access
- **Loop over K-tiles**: `for kk in range(num_k_tiles): tile = ct.load(A, (m_idx, kk), shape=...)`

---
name: cutile-attention-kernels
description: cuTile patterns specific to fused multi-head attention, flash decode, and attention sink kernels
category: coding
---

## cuTile Attention Kernel Patterns

- **FMHA tiling**: Factor attention into a reusable `_impl` device function that accepts block indices explicitly, enabling the same logic for both standard and attention-sink variants.
- **Online softmax**: Initialize `m_i = ct.full((TILE_M, 1), float('-inf'), ...)` and `l_i = ct.full((TILE_M, 1), 0.0, ...)` accumulators; update per K-tile with running max and sum correction.
- **Causal masking**: Use `ct.Constant[bool]` for `CAUSAL` flag to compile-time branch; generate causal masks from tile-local `ct.arange` offsets.
- **Grouped Query Attention (GQA)**: Multiple Q heads share one KV head via `NUM_Q_HEAD_PER_KV`. Load Q as `(1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM)`, then `ct.reshape()` to 2D before MMA.
- **Split-K parallelism** (flash decode): Split the KV sequence across grid dimension, each CTA computes partial softmax; reduce across splits afterward.
- **Attention sinks**: Load sink count as scalar via `ct.load(Sinks, ...).item()`; handle initial sink tokens separately before processing the main KV range.
- **TMA usage**: Always pass `allow_tma=True` and explicit `order` for Q/K/V loads to enable Tensor Memory Accelerator for efficient global→shared transfers.
- **Occupancy hint**: `@ct.kernel(occupancy=2)` for 2 concurrent CTAs per SM is typical for attention kernels.

---
name: cutile-elementwise-and-1d-kernels
description: cuTile patterns for elementwise and 1D operations: gather/scatter instead of load/store, explicit math functions with parameters, and proper type casting
category: coding
---

## cuTile Elementwise & 1D Kernel Patterns

- **Element access**: `ct.gather(tensor, (row_idx, col_offsets), check_bounds=True)` and `ct.scatter(tensor, (row_idx, col_offsets), value, check_bounds=True)`
- **Explicit math ops**: `ct.add(a, b)`, `ct.mul(a, b)`, `ct.truediv(a, b, flush_to_zero=True, rounding_mode='rn')` — no implicit operator overloading assumed
- **Type casting**: `ct.astype(tensor, target_dtype)` for explicit conversions
- **No built-in `ct.exp`/`ct.rand`**: Implement math functions manually or use available primitives
- **Bitwise ops**: `ct.bitwise_xor(a, b)` available for manual PRNG or hash functions
- **Index pattern**: `offsets = ct.arange(TILE_SIZE, dtype=ct.int32)` then `ct.gather(X, (block_idx, offsets))`
- **Bounds checking**: Always use `check_bounds=True` or `padding_value` instead of manual mask computation
- **Constants**: Annotate compile-time params as `ct.Constant[int]` or `ct.Constant[bool]` in kernel signatures

---

# Critical Validation Rules (learned from evaluation failures)

## Source Reference Validation

Every cuTile solution MUST use actual cuTile APIs. The evaluator validates that your solution contains required source references — specific ct.* function calls. Solutions that produce correct results using pure PyTorch will FAIL validation.

**Required in every solution:**
- `@ct.kernel` decorator on at least one function
- `ct.launch()` to invoke the kernel
- At least one of: `ct.load`, `ct.gather`, `ct.store`, `ct.scatter` for memory access

**Before submitting, verify your solution.py contains these patterns:**
```python
import cuda.tile as ct

@ct.kernel
def my_kernel(...):
    # Must use ct.* ops inside kernel
    data = ct.load(...)  # or ct.gather(...)
    ct.store(...)        # or ct.scatter(...)

ct.launch(stream, grid, my_kernel, args)
```

**Anti-pattern:** Writing a solution using only `torch.matmul`, `torch.nn.functional`, or Python loops. Even if the output is numerically correct, it will fail source reference validation.

## Numerical Precision

For matmul, normalization, and reduction operations:
- Always accumulate in `ct.float32` regardless of input dtype
- Use `ct.full((M, N), 0.0, dtype=ct.float32)` for accumulators
- Cast back to input dtype only at final output: `ct.astype(result, input_dtype)`
- For layer norm backward pass, tight tolerances (1e-6) require float32 throughout

## Self-Validation Checklist

Before submitting, run this mental checklist:
1. Does solution.py contain `@ct.kernel`? If not, rewrite using cuTile DSL.
2. Does solution.py contain `ct.launch()`? If not, add kernel launch.
3. Are all computations using ct.* ops inside the kernel (not PyTorch)?
4. Are accumulators in float32 for numerical stability?
5. Does the output shape exactly match what the problem spec requires?
6. Is the tensor on CUDA device (not CPU)?
