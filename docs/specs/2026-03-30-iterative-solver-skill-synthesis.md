# Iterative Solver + Skill Synthesis from Debug Journeys

**Date:** 2026-03-30
**Status:** Draft
**Author:** Kaix + Claude
**Sub-project:** 1 of 2 (this: solver + extractor; next: problem collector for more sources)

## Problem Statement

Our previous skill evolution approaches showed:
- Method 1 (direct extraction): +0 tasks — LLM already knows generic patterns
- Method 2 (fix-based learning): +3 tasks — positive patterns from successful fixes help
- Method 3 (prescriptive rules): -1 task — rules over-constrain the agent

The missing ingredient: **real compile/runtime error feedback**. When the agent has test access (v1), it scores 89.6%. Without tests (v2), it scores 66.7%. The 23pp gap comes from the agent's inability to debug its own code without real error messages.

**Insight:** If we let the agent debug cuTile kernels with real errors during skill learning, then extract skills from those debugging sessions, the skills should help the agent when it works blind on the benchmark.

## Approach

1. **Iterative Solver:** OpenCode agent in Docker solves TileGym kernels with real test access and 48 tool calls — capturing the full debug journey (errors seen, fixes applied)
2. **Skill Extractor:** Parse agent traces → extract error→fix pairs → LLM synthesizes into teaching notes → feed through MetaClaw skill evolution pipeline
3. **Benchmark:** Evaluate synthesized skills on compute-eval (tests hidden, 24 tool calls) — fair comparison

**Key principle:** Learn with full feedback (tests visible, 48 calls), evaluate blind (tests hidden, 24 calls).

---

## System Architecture

```
Problem Source (TileGym 22 kernels)
  │
  ▼
Iterative Solver (OpenCode in Docker, tests visible, 48 tool calls)
  │
  ├─ Per problem: read problem → write cuTile → run test → see error → fix → repeat
  ├─ Produces: solution + agent trace (all tool calls, errors, fixes)
  │
  ▼
Skill Extractor
  │
  ├─ Stage 1: Parse traces → extract error→fix pairs (programmatic)
  ├─ Stage 2: LLM synthesizes pairs into teaching notes
  ├─ Stage 3: Feed through ConversationSignalDetector → SkillEvolver
  │
  ▼
Evolved Skills (merged into cutile-python-synthesized/)
  │
  ▼
Benchmark (compute-eval-synth v2, tests HIDDEN, 24 tool calls)
  │
  ▼
Results: pass rate comparison vs baseline
```

---

## Component 1: Iterative Solver

### Workspace Setup (per problem)

For each of 22 TileGym kernels, create a workspace directory:

```
workspace/<kernel_name>/
├── problem-spec.yaml              # What to implement
├── test/test_<kernel>.py           # Real TileGym test (VISIBLE for learning)
├── context/                        # Context files from TileGym (if any)
└── cutile-python/                  # Skill directory (base + any evolved skills)
    └── SKILL.md
```

The problem spec is generated from TileGym's test file — it describes what the kernel should do without giving the reference implementation.

### Solver Execution

Uses OpenCode CLI inside the `local/opencode-agent:cutile-synthv2` Docker container:

```bash
docker run --rm --gpus all \
  -v <workspace>:/testbed:rw \
  -v <skill_dir>:/testbed/.opencode/skill/cutile-python:ro \
  -e CUDA_TILE_CACHE_DIR=/tmp/cutile-cache \
  -e OPENAI_API_KEY=$API_KEY \
  local/opencode-agent:cutile-synthv2 \
  opencode run -m nvidia/aws/anthropic/bedrock-claude-opus-4-6 \
    --dir /testbed \
    --max-steps 48 \
    "<prompt>"
```

**Prompt:**
```
You are solving a cuTile kernel implementation problem.
Read problem-spec.yaml and the cuTile skill first.
Write solution.py that passes the test in test/.
Run the test to verify your solution. If it fails, read the error and fix your code.
Keep iterating until the test passes.
```

**Configuration:**
- Model: Claude Opus 4.6 (`aws/anthropic/bedrock-claude-opus-4-6`)
- API: `https://inference-api.nvidia.com/v1`
- Max steps: 48 (configurable, default 48 for learning)
- Tests: visible (mounted into container)
- GPU: B200 (required for cuTile)
- OpenCode config: `OPENCODE_CONFIG_CONTENT` with nvidia provider

### Why custom wrapper instead of synth-v2

synth-v2 (`generate_synth.py solve`) hides test files by design. Rather than patching synth-v2, we write our own solver that:
1. Creates the workspace with tests included
2. Runs OpenCode directly via `docker run`
3. Collects results and traces

This avoids modifying upstream code and gives full control over workspace contents.

### Output per problem

```
results/<kernel_name>/
├── solution.py              # Final solution
├── trace.jsonl              # Full agent trace
├── result.json              # {passed: bool, num_steps: N, errors_seen: [...]}
```

### Expected results

~15-18 of 22 kernels solved. Complex attention/MLA kernels (1270+ lines) will likely fail — they exceed the agent's context and step budget.

---

## Component 2: Skill Extractor

### Stage 1: Trace Parser (programmatic)

Parse each `trace.jsonl` from successful solves to extract error→fix pairs.

**Trace format** (OpenCode `trace.jsonl`):
```jsonl
{"type": "step_start", ...}
{"type": "tool_use", "part": {"toolName": "write", "input": {"path": "solution.py", "content": "..."}}}
{"type": "tool_use", "part": {"toolName": "bash", "input": {"command": "python -m pytest test/"}}}
{"type": "tool_result", "part": {"output": "FAILED: RuntimeError: ct.mma expects 2D tiles"}}
{"type": "text", "part": {"text": "I need to reshape the tile before MMA..."}}
{"type": "tool_use", "part": {"toolName": "write", "input": {"path": "solution.py", "content": "...fixed code..."}}}
{"type": "tool_use", "part": {"toolName": "bash", "input": {"command": "python -m pytest test/"}}}
{"type": "tool_result", "part": {"output": "PASSED"}}
```

**Extraction logic:**

```python
for each trace:
    find sequences: write(code) → run(test) → error → write(fixed_code) → run(test) → pass
    extract: {
        "error_message": "RuntimeError: ct.mma expects 2D tiles",
        "code_before": <code that caused error>,
        "code_after": <code that fixed it>,
        "agent_reasoning": <text between error and fix>,
        "kernel_name": "matmul"
    }
```

### Stage 2: LLM Synthesis (teaching notes)

Group similar error→fix pairs across kernels by error pattern (keyword clustering, same as `CorrectionClusterer`).

For each group:

```python
prompt = f"""
While debugging cuTile kernels, these errors and fixes were encountered:

{formatted_error_fix_pairs}

Write a concise teaching note (3-5 sentences) starting with "Remember this:"
that captures the reusable pattern. Include:
- What error to watch for
- Why it happens
- How to fix it
Be specific about cuTile API usage (ct.* functions).
"""
```

Example output:
```
Remember this: when using ct.mma for tensor core operations, both input tiles
must be 2D. If you load a 3D tile with ct.load(A, index=(batch, row, col),
shape=(1, M, K)), reshape it to 2D before MMA: ct.reshape(tile, (M, K)).
The batch dimension should be handled by the grid, not the tile shape.
```

### Stage 3: Skill Evolution Pipeline

Teaching notes flow through the existing MetaClaw pipeline:
1. `ConversationSignalDetector` — detects "Remember this:" → `explicit_save`
2. `SignalAggregator` — buffers all signals
3. `SkillEvolver` — synthesizes across teaching notes, creates/updates skills
4. `SkillManager` — writes SKILL.md files to disk

### Output

```
synthesis-results/
├── traces/                          # Raw agent traces per kernel
├── error_fix_pairs.jsonl            # Extracted pairs
├── teaching_notes.jsonl             # LLM-generated notes
├── evolved-skills/                  # Individual skill dirs
│   ├── cutile-tile-reshaping/SKILL.md
│   ├── cutile-precision-patterns/SKILL.md
│   └── cutile-memory-access/SKILL.md
└── cutile-python-synthesized/       # Merged skill dir for benchmark
    └── SKILL.md                     # Base + all evolved skills appended
```

---

## Component 3: Benchmark Evaluation

Uses the existing `cutile-synth-eval-kit` pipeline (v2, tests hidden):

```bash
# Same as v2-run-baseline.sh but with synthesized skills
export CUTILE_SKILL_DIRS="$WORKDIR/eval/cutile/synthesis-results/cutile-python-synthesized"
crun -q 'gpu.product_name=*B200*' --gpus 1 -t 8:00:00 -b runs/v2-run-synthesized.sh
```

**Conditions compared:**
```
Method                          Tests visible?  Tool calls  Skills
─────────────────────────────────────────────────────────────────
Baseline                        Hidden          24          cutile-python only
Method 2 (fix-based)            Hidden          24          + fix-based skills
Synthesized (debug journeys)    Hidden          24          + debug journey skills
```

---

## File Structure

```
eval/cutile/
├── # Existing (unchanged)
├── run_eval.py
├── phase_a_study.py
├── phase_b_practice.py
├── kernel_tester.py
├── llm_client.py
├── tilegym_loader.py
│
├── # New
├── iterative_solver.py          # OpenCode-in-Docker solver with tests visible
├── trace_parser.py              # Parse traces → error→fix pairs
├── skill_synthesizer.py         # LLM synthesis + MetaClaw pipeline integration
├── run_skill_synthesis.py       # Main entry point
│
└── Dockerfile.tilegym           # Existing (reuse)
```

---

## CLI

```bash
# Step 1: Solve + Extract (B200 required)
python eval/cutile/run_skill_synthesis.py \
  --tilegym-dir TileGym \
  --results-dir eval/cutile/synthesis-results \
  --base-skill skills/cutile-python \
  --model aws/anthropic/bedrock-claude-opus-4-6 \
  --max-steps 48

# Step 2: Evaluate (B200 required, tests hidden)
crun -q 'gpu.product_name=*B200*' --gpus 1 -t 8:00:00 \
  -b runs/v2-run-synthesized.sh
```

Step 1 can also be wrapped in a single B200 batch script.

---

## Expected Results

```
Method                          Pass    pass@1    Delta vs Baseline
────────────────────────────────────────────────────────────────────
Baseline                        32/48   66.7%     —
Method 2 (fix-based)            35/48   72.9%     +3
Synthesized (debug journeys)    38+/48  79%+      +6 (target)
```

Target is based on: the debug journey skills should capture the real error patterns that cause the 13 test-gap failures, specifically:
- Source reference validation: skills will show "use ct.kernel + ct.launch" from actual errors
- Numerical precision: skills will show "use float32 accumulators" from actual tolerance failures
- Shape mismatches: skills will show correct tensor dimension patterns

---

## Infrastructure Requirements

### Docker Images
- `local/opencode-agent:cutile-synthv2` — for iterative solving (already built)
- `local/compute-eval-tilegym:13.1.0` — for TileGym test execution (already built)
- `local/compute-eval-python:13.1.0` — for benchmark evaluation (already built)

All saved as tars in `runs/`.

### B200 GPU
- Step 1 (solve 22 kernels): ~1.5 hours with MAX_WORKERS=1
- Step 2 (benchmark 48 problems): ~2 hours

### LLM API
- Solve: ~22 × 48 tool calls × ~$0.05/call = ~$50
- Synthesis: ~30 LLM calls = ~$1.50
- Benchmark: handled by v2 pipeline

---

## Differences from Previous Approaches

| | Method 2 (previous) | Synthesized (this) |
|---|---|---|
| Problem source | TileGym kernels | Same |
| Agent | Direct LLM call | OpenCode in Docker (agentic) |
| Test access | LLM comparison (broken) → then real tests | Real tests from start |
| Error feedback | None (single attempt) or reference-based | Real compile/runtime errors |
| Fix attempts | 1 blind + 3 with reference | Up to 48 tool calls with error feedback |
| Skill source | Diff between failed/fixed code | Full debug journey (errors + reasoning + fixes) |
| Expected improvement | +3 tasks | +6 tasks (target) |
