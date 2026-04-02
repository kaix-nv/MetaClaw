# cuTile Skill Synthesis via Autoresearch-Style Experiment Loop

**Date:** 2026-04-02
**Status:** Draft
**Author:** Kaix + Claude
**Supersedes:** `2026-03-30-iterative-solver-skill-synthesis.md` (same goal, better structure)

## Problem Statement

Previous approaches to cuTile skill evolution showed:
- Direct extraction: +0 (LLM already knows generic patterns)
- Fix-based learning: +3 tasks (positive patterns from fixes help)
- Prescriptive rules: -1 (over-constrains the agent)
- Our iterative solver had Docker mount issues and OpenCode configuration problems

The autoresearch framework provides a proven pattern: give the agent a structured `program.md`, let it modify code, run tests, keep/discard, and iterate autonomously.

## Approach

For each of 22 TileGym kernels:
1. Create a workspace with `program.md` + problem spec + test file + cuTile skill
2. Launch OpenCode in Docker (one session per problem)
3. Agent follows `program.md`: write solution → test → fix → repeat until pass
4. Git tracks every iteration (commit per attempt)
5. After all problems: parse git histories → extract error→fix pairs → synthesize skills
6. Evaluate synthesized skills on compute-eval benchmark (tests hidden)

**Key principle from autoresearch:** The agent manages its own experiment loop via `program.md`. We don't build a Python orchestrator that calls the LLM in a loop — the agent IS the loop.

---

## Architecture

```
For each of 22 TileGym kernels:

  Workspace:
    workspace/<kernel>/
    ├── program.md              # Agent instructions (same for all kernels)
    ├── problem-spec.yaml       # What to implement
    ├── test/test_<kernel>.py   # Real test (VISIBLE)
    ├── .opencode/skill/cutile-python/  # cuTile skill
    └── solution.py             # Agent writes/modifies this

  Agent (OpenCode in Docker):
    reads program.md → follows the loop → commits each attempt
    ↓
  Output:
    ├── solution.py             # Final code
    ├── .git/                   # Full iteration history
    └── results.log             # Pass/fail + error messages per attempt

Post-processing (across all 22 kernels):
  Parse git diffs + results.log → error→fix pairs → LLM synthesis → skills
```

---

## program.md

The single instruction document that drives the agent:

```markdown
# cuTile Kernel Implementation

## Goal
Implement a cuTile kernel that passes the test in `test/`.

## Setup
1. Read the cuTile skill in `.opencode/skill/cutile-python/SKILL.md`
2. Read `problem-spec.yaml` to understand what to implement
3. Read the test file in `test/` to understand expected behavior
4. Initialize git: `git init && git add -A && git commit -m "initial"`

## Experiment Loop
Repeat until the test passes or you've made 15 attempts:

1. Write or modify `solution.py` with your cuTile implementation
2. Commit: `git add solution.py && git commit -m "attempt N: <brief description>"`
3. Run the test: `cd /testbed && python -m pytest test/ -x -v 2>&1 | tee -a results.log`
4. Check the result:
   - If PASSED: commit final state and stop
   - If FAILED: read the error message carefully, understand what went wrong,
     and go back to step 1 with a fix

## Rules
- Only create/modify `solution.py` — do not modify files in `test/`
- Use the cuTile API: `import cuda.tile as ct`
- Every solution must use `@ct.kernel` and `ct.launch()`
- Use float32 accumulators for numerical stability
- Always commit before testing (so we can track your iterations)
- Read error messages carefully — they tell you exactly what's wrong

## Output
When done (pass or max attempts), append to results.log:
```
FINAL: <PASS|FAIL> attempts=<N> kernel=<kernel_name>
```
```

---

## Workspace Setup

For each kernel, the orchestrator creates a workspace:

```python
def create_workspace(kernel_name, workspace_dir, tilegym_dir, skill_dir):
    # 1. Copy program.md (same for all kernels)
    copy("program.md", workspace_dir / "program.md")

    # 2. Write problem spec
    write_yaml(workspace_dir / "problem-spec.yaml", {
        "kernel": kernel_name,
        "description": f"Implement the {kernel_name} cuTile kernel",
    })

    # 3. Copy test file (VISIBLE to agent)
    copy(tilegym_dir / f"tests/ops/test_{kernel_name}.py",
         workspace_dir / "test" / f"test_{kernel_name}.py")
    # Also copy test utilities (common.py, conftest.py)

    # 4. Copy cuTile skill
    copytree(skill_dir, workspace_dir / ".opencode/skill/cutile-python")
```

## Solver Execution

One Docker container per kernel:

```bash
docker run --rm --gpus all \
  --entrypoint "" \
  -v <workspace>:/testbed:rw \
  -e OPENCODE_CONFIG_CONTENT=<json> \
  -e OPENAI_API_KEY=$API_KEY \
  -e CUDA_TILE_CACHE_DIR=/tmp/cutile-cache \
  -e XDG_DATA_HOME=/tmp/xdg-data \
  -e XDG_CONFIG_HOME=/tmp/xdg-config \
  -e XDG_STATE_HOME=/tmp/xdg-state \
  -w /testbed \
  local/opencode-agent:cutile-synthv2 \
  opencode run -m nvidia/aws/anthropic/bedrock-claude-opus-4-6 \
    --dir /testbed \
    "Follow the instructions in program.md"
```

**Timeout:** 10 minutes per kernel (enough for ~15 attempts at ~30s each)

---

## Post-Processing: Skill Extraction

After all 22 kernels are attempted:

### Stage 1: Parse git histories

For each workspace that has a passing solution:

```bash
cd workspace/<kernel>
git log --oneline                    # List all attempts
git diff HEAD~1 HEAD -- solution.py  # Diff between last fix and pass
```

Extract error→fix pairs:
```python
for each consecutive pair of commits (attempt_N, attempt_N+1):
    diff = git_diff(attempt_N, attempt_N+1, "solution.py")
    error = grep("FAILED|Error|Traceback", results.log, after=attempt_N)
    if error and diff:
        pairs.append({
            "error_message": error,
            "code_before": git_show(attempt_N, "solution.py"),
            "code_after": git_show(attempt_N+1, "solution.py"),
            "kernel_name": kernel_name,
        })
```

### Stage 2: LLM synthesis (same as before)

Group pairs → LLM generates teaching notes → MetaClaw pipeline → skills

### Stage 3: Merge

Copy base `cutile-python` skill → append evolved skills → `cutile-python-synthesized/`

---

## File Structure

```
eval/cutile/
├── program.md                   [NEW]  Agent instructions (autoresearch-style)
├── autoresearch_solver.py       [NEW]  Orchestrator: workspace setup + Docker launch
├── git_trace_parser.py          [NEW]  Parse git history → error→fix pairs
├── skill_synthesizer.py         [EXISTING]  LLM synthesis (reuse)
├── run_skill_synthesis.py       [MODIFY]  Updated entry point
├── tilegym_loader.py            [EXISTING]  Load kernels (reuse)
├── llm_client.py                [EXISTING]  LLM API (reuse)
├── skill_writer.py              [EXISTING]  Merge skills (reuse)
```

**Changes from previous iterative solver:**
- `iterative_solver.py` → replaced by `autoresearch_solver.py` (simpler, program.md-driven)
- `trace_parser.py` → replaced by `git_trace_parser.py` (parses git diffs, not OpenCode traces)
- `program.md` → new (the agent instruction document)
- `skill_synthesizer.py` → reused as-is

---

## CLI

```bash
# Full pipeline: solve + extract + merge (B200 required)
python eval/cutile/run_skill_synthesis.py \
  --tilegym-dir TileGym \
  --results-dir eval/cutile/synthesis-results \
  --base-skill skills/cutile-python \
  --model aws/anthropic/bedrock-claude-opus-4-6 \
  --timeout 600

# Then benchmark (B200 required, tests hidden)
crun -q 'gpu.product_name=*B200*' --gpus 1 -t 8:00:00 \
  -b runs/v2-run-synthesized.sh
```

---

## Expected Results

```
Method                          Pass    pass@1    Delta vs Baseline
────────────────────────────────────────────────────────────────────
Baseline                        32/48   66.7%     —
Method 2 (fix-based)            35/48   72.9%     +3
Autoresearch synthesis          38+/48  79%+      +6 (target)
```

---

## Differences from Previous Iterative Solver

| | Previous (iterative_solver.py) | New (autoresearch-style) |
|---|---|---|
| Agent control | Python orchestrator calls OpenCode | Agent follows program.md autonomously |
| Iteration tracking | Parse OpenCode traces (complex format) | Parse git history (simple diffs) |
| Error extraction | Regex on trace.jsonl | `git diff` + grep results.log |
| Commit strategy | No commits during solving | Every attempt committed |
| Complexity | Custom Docker mount logic, entrypoint issues | Standard workspace mount, agent manages itself |
| Debugging | Opaque traces | `git log` shows full history |
