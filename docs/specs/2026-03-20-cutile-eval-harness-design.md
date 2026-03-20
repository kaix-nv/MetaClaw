# cuTile Evaluation Harness for Conversation-Driven Skill Evolution

**Date:** 2026-03-20
**Status:** Draft (v2 — TileGym-based)
**Author:** Kaix + Claude

## Problem Statement

We built conversation-driven skill evolution into MetaClaw (skills extracted from user corrections, iterated via feedback). We need a concrete, measurable evaluation: does the system actually improve an agent's performance through conversation-driven skill learning?

cuTile kernel coding provides the test domain. **TileGym** (31 cuTile kernel implementations + 100+ test cases) serves as the training ground for skill evolution. The **compute-eval benchmark** (48 cuTile problems) serves as the held-out test set.

## Design Goals

- Exercise the **full conversation pipeline**: `ConversationSignalDetector` → `SignalAggregator` → `SkillEvolver` → `SkillManager`
- **Train/test separation**: learn skills from TileGym, evaluate on compute-eval
- Two-phase skill extraction: study code (Phase A) + solve-and-check (Phase B)
- Measure pass rate improvement on compute-eval with skill attribution
- Configurable rounds (default: single, option for multi-round convergence)
- Locally testable (skill evolution) without GPU/cluster

## Approach

Python orchestrator inside `MetaClaw/eval/cutile/` with three phases:

1. **Phase A (Study):** LLM reads TileGym kernel implementations, generates teaching summaries → explicit skill extraction via conversation pipeline
2. **Phase B (Practice):** Agent attempts to re-implement TileGym kernels from test specs, gets corrected against reference → implicit correction-driven skill refinement
3. **Phase C (Evaluate):** Evolved skills injected into compute-eval benchmark solve → measure pass rate vs baseline

---

## Architecture Overview

```
MetaClaw/eval/cutile/
├── run_eval.py              # Main orchestrator (entry point)
├── phase_a_study.py         # Read TileGym code, generate teaching summaries
├── phase_b_practice.py      # Solve-and-check against TileGym references
├── phase_c_evaluate.py      # Run compute-eval benchmark with evolved skills
├── failure_analyzer.py      # Parse eval results, extract error context
├── correction_simulator.py  # Generate corrections from reference comparison
├── skill_writer.py          # Write/merge skills for injection
├── report.py                # Per-round and cross-round comparison reports
└── README.md                # Usage instructions
```

**Data sources:**
- TileGym: `/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw/TileGym/`
  - Kernel implementations: `src/tilegym/ops/cutile/` (31 files)
  - Test files: `tests/ops/test_*.py` (20 files, ~100+ parametrized cases)
  - Operation interfaces: `src/tilegym/ops/ops.py`
- compute-eval: via `cutile-eval-kit/` shell wrappers (48 problems, B200 Docker eval)

---

## Phase A: Study TileGym Code → Extract Initial Skills

The LLM reads each TileGym kernel implementation and generates a "teaching summary" — a natural language description of the cuTile patterns used. These summaries flow through the full conversation signal detection pipeline.

### Input per kernel

```python
kernel_source = read_file("src/tilegym/ops/cutile/softmax.py")
test_source = read_file("tests/ops/test_softmax.py")
```

### Teaching summary generation

```python
teaching_prompt = f"""
You are a cuTile expert writing a teaching note for a student who will
implement GPU kernels using the cuTile Python DSL.

Study this kernel implementation and its tests:

Implementation:
```python
{kernel_source[:2000]}
```

Tests:
```python
{test_source[:1000]}
```

Write a concise teaching note (3-5 sentences) explaining the key cuTile
patterns and API usage in this kernel. Start with "Remember this:" to
make it clear this is a skill to save. Be specific about ct.* API calls,
tile sizes, and hardware features (TMA, tensor cores) used.
"""

teaching_message = call_llm(teaching_prompt)
# e.g. "Remember this: when implementing softmax in cuTile, use ct.load
#        with use_tma=True for rows larger than 1024 elements. Use the
#        chunked approach for smaller rows. Always compute ct.max first
#        to subtract the maximum before ct.exp to avoid overflow."
```

### Feed through conversation pipeline

```python
turn_data = {
    "session_id": f"tilegym-study-{kernel_name}",
    "turn_num": 1,
    "user_message": teaching_message,      # "Remember this: ..." → explicit_save
    "assistant_response": kernel_source,    # The kernel being studied
    "active_skills": [],
}
signals = detector.detect(turn_data)       # explicit_save detected
aggregator.add(signals)
# explicit_save triggers immediately → evolve
```

### Expected output

After studying all 31 kernels: **~10-20 initial skills** covering cuTile patterns (API usage, hardware features, algorithmic patterns). The SkillEvolver synthesizes across multiple teaching summaries, merging related patterns into general skills rather than creating one skill per kernel.

---

## Phase B: Solve-and-Check → Refine Skills

The agent attempts to implement each TileGym kernel given only its test file as specification. When the attempt differs from the reference, a correction is generated and fed through the conversation pipeline.

### Problem format

For each of the 31 TileGym kernels:

**Given to the agent:**
- Test file (e.g., `tests/ops/test_softmax.py`) — defines inputs, expected outputs, tolerances
- Current evolved skills (from Phase A + previous iterations)

**NOT given:**
- The reference implementation (`src/tilegym/ops/cutile/softmax.py`)

**Agent task:** Write the cuTile kernel implementation that passes the tests.

### Solve attempt

```python
solve_prompt = f"""
You are solving a cuTile kernel implementation problem.

Test specification:
```python
{test_source}
```

Write the cuTile kernel implementation that passes these tests.
Use the ct.* API (ct.kernel, ct.launch, ct.load, ct.store, etc.).
Write only the implementation code.
"""

agent_solution = call_llm(solve_prompt)
```

### Compare against reference

```python
reference = read_file(f"src/tilegym/ops/cutile/{kernel_name}.py")

comparison_prompt = f"""
Compare this student's cuTile kernel attempt with the reference implementation.

Student attempt:
```python
{agent_solution[:1000]}
```

Reference (correct):
```python
{reference[:1000]}
```

If the student's code has errors or uses wrong patterns, write a short
correction (2-3 sentences). Start with "No, that's wrong." and explain
what should be done differently. Be specific about cuTile API usage.

If the student's code is essentially correct, respond with just "CORRECT".
"""

comparison = call_llm(comparison_prompt)
```

### Correction flow (only when wrong)

```python
if comparison.strip() != "CORRECT":
    turn_data = {
        "session_id": f"tilegym-practice-round{round}-{kernel_name}",
        "turn_num": 1,
        "user_message": comparison,             # "No, that's wrong. ..."
        "assistant_response": agent_solution,    # The wrong attempt
        "active_skills": current_skill_names,
    }
    signals = detector.detect(turn_data)         # implicit_correction detected
    aggregator.add(signals)
```

### Evolution trigger

After processing all 31 kernels (or when correction threshold met):

```python
if aggregator.should_evolve():
    consumed = aggregator.consume()
    actions = await evolver.evolve(consumed, skill_manager.skills)
    # SkillEvolver sees all corrections together → synthesizes patterns
    # e.g., "5 corrections about using built-in ct.* ops instead of manual loops"
    #      → creates skill "prefer-builtin-ct-ops"
    for action in actions:
        if action.action == "create":
            skill_manager.add_skill(action.skill)
        elif action.action == "update":
            skill_manager.update_skill(action.skill["name"], action.skill)
```

### Multi-round iteration (configurable)

```
Round 1: Agent attempts all 31 kernels with Phase A skills only
  → corrections extracted → skills evolved
Round 2: Agent re-attempts failed kernels with Round 1 skills
  → corrections extracted → skills further refined
Round N: Until no new corrections or --rounds limit reached
```

### Correction simulation: Avoiding explicit detection mismatch

The `ConversationSignalDetector` skips implicit detection when an explicit pattern matches. The LLM correction prompt must avoid explicit-save trigger words.

**Constraint in correction prompt:** The prompt says "Start with 'No, that's wrong.'" which ensures the implicit correction pattern matches. The prompt does NOT use "always", "never", "from now on", "remember this".

**Fallback:** If the LLM-generated correction doesn't trigger detection:
```python
if not detector.detect(turn_data):
    turn_data["user_message"] = f"No, that's wrong. {comparison}"
    signals = detector.detect(turn_data)
```

---

## Phase C: Evaluate on Compute-Eval Benchmark

All evolved skills from Phases A+B are tested on the held-out 48 cuTile problems.

### Skill injection

`skill_writer.py` concatenates base cuTile skill + all evolved skills into a single flat markdown file (the eval kit's generator accepts one `--skill-md` file):

```python
def write_merged_skill(base_skill_path, evolved_skills, output_path):
    """Concatenate base cuTile skill + all evolved skills into one file."""
    parts = [Path(base_skill_path).read_text()]
    parts.append("\n\n---\n\n# Evolved Skills (learned from TileGym)\n")
    for skill in evolved_skills:
        parts.append(f"\n## {skill['name']}\n_{skill['description']}_\n\n{skill['content']}\n")
    Path(output_path).write_text("\n".join(parts))
```

### Evaluation execution

```bash
# Baseline (no evolved skills):
env MODEL=$model SKILL_MD=$base_skill \
  bash $eval_kit_dir/scripts/compute-eval-run-cutile-opencode-gpt52.sh

# With evolved skills:
env MODEL=$model SKILL_MD=$merged_skill \
  bash $eval_kit_dir/scripts/compute-eval-run-cutile-opencode-gpt52.sh
```

### Reading results

The eval script writes graded results to `$RUN_ROOT/eval-output/` with a dynamic filename `*-graded-solutions.jsonl`. The orchestrator finds it via:

```python
import glob
graded = sorted(glob.glob(f"{run_root}/eval-output/*-graded-solutions.jsonl"))[-1]
```

Agent solutions are in `$RUN_ROOT/workspaces/<task_id>/` and packed into `$OUTPUT_DIR/<release>-<model>-solutions.tar.gz`.

---

## Reporting and Skill Attribution

### Per-round data

```json
{
  "round": 1,
  "phase_a_skills": 15,
  "phase_b_corrections": 18,
  "phase_b_skills_evolved": 5,
  "total_skills": 20,
  "compute_eval_pass_rate": "28/48",
  "per_task": {
    "cutile/0": {"status": "passed", "active_skills": ["softmax-tma-loading", "prefer-builtin-ct-ops"]},
    "cutile/1": {"status": "failed", "error_type": "compile", "error": "..."}
  }
}
```

### Cross-round comparison

```
=== cuTile Skill Evolution Report ===

Phase A: 15 skills extracted from studying 31 TileGym kernels
Phase B: 5 additional skills from solve-and-check corrections

       Pass  Fail  Skip  Source
Base   20    25     3    cutile-minimal-skill.md only
+A     25    20     3    + Phase A skills (study)
+A+B   28    17     3    + Phase B skills (practice)
+A+B×2 31    14     3    + Phase B round 2 refinement

Flips (fail→pass):
  Base→+A:    cutile/3, cutile/7, cutile/12, cutile/22, cutile/31
  +A→+A+B:   cutile/5, cutile/15, cutile/40
  +A+B→+A+B×2: cutile/19, cutile/33, cutile/44

Skill Attribution:
  softmax-tma-loading          → cutile/3, cutile/12 (Phase A)
  prefer-builtin-ct-ops        → cutile/7, cutile/22, cutile/5 (Phase A+B)
  tile-shape-tensor-cores      → cutile/31, cutile/15 (Phase A+B)
  reduction-with-ct-sum        → cutile/40, cutile/19 (Phase B)
```

### Output files

```
results/
├── phase-a/
│   ├── teaching-summaries.jsonl    # LLM-generated summaries per kernel
│   └── evolved-skills/             # Skills from studying code
├── phase-b/
│   ├── round-1/
│   │   ├── attempts.jsonl          # Agent solutions per kernel
│   │   ├── corrections.jsonl       # Generated corrections
│   │   └── evolved-skills/         # Skills from corrections
│   └── round-2/ ...
├── phase-c/
│   ├── baseline/
│   │   └── eval-results.jsonl      # No evolved skills
│   ├── with-skills/
│   │   └── eval-results.jsonl      # With evolved skills
│   └── comparison.json             # Per-task diff
└── summary.txt                     # Human-readable report (table above)
```

---

## Configuration and CLI

```bash
python eval/cutile/run_eval.py \
  --rounds 2 \
  --tilegym-dir /path/to/TileGym \
  --eval-kit-dir /path/to/cutile-eval-kit \
  --results-dir eval/cutile/results \
  --model azure/openai/gpt-5.2 \
  --evolver-model gpt-5.2 \
  --base-skill /path/to/cutile-minimal-skill.md \
  --task-ids ""
```

**Phases can run independently:**

```bash
# Phase A only (local, no GPU):
python eval/cutile/run_eval.py --phase a --tilegym-dir ...

# Phase B only (local, no GPU):
python eval/cutile/run_eval.py --phase b --tilegym-dir ... --skills-from results/phase-a/

# Phase C only (needs B200):
python eval/cutile/run_eval.py --phase c --eval-kit-dir ... --skills-from results/phase-b/round-2/
```

### Local testing mode

```bash
python eval/cutile/run_eval.py \
  --local-test \
  --mock-eval-results path/to/previous/eval-results.jsonl
```

Skips Phase C subprocess calls. Uses previous eval results. Tests the full Phase A → Phase B → reporting pipeline without GPU/cluster.

### Task ID handling

When `--task-ids` is empty or omitted, all tasks are run. The orchestrator passes `TASK_IDS=""` (empty string) to subprocess — this is the eval kit's convention for "all tasks". Never pass `None`.

---

## Integration with Existing Systems

### MetaClaw components used

All imports from `metaclaw.*`:
- `ConversationSignalDetector` — detects explicit_save (Phase A) and implicit_correction (Phase B)
- `SignalAggregator` + `SkillEvolutionConfig` — buffers signals, triggers evolution
- `SkillEvolver` — LLM-based skill synthesis from signals
- `SkillManager` — stores/retrieves/updates skills on disk
- `SkillAction` — create/update/deprecate/link actions

`SkillEvolutionConfig` is imported from `metaclaw.signal_aggregator` (not `metaclaw.config`). The `MetaClawConfig.skill_evolution_config()` helper handles conversion from flat config fields.

### Eval kit scripts called

Via subprocess with env var overrides:
- `cutile-eval-kit/scripts/compute-eval-run-cutile-opencode-gpt52.sh` — solve + eval
- `cutile-eval-kit/scripts/run_b200_local_compute_eval_evalonly.sh` — eval-only rerun

### TileGym files read (not modified)

- `TileGym/src/tilegym/ops/cutile/*.py` — reference implementations (31 files)
- `TileGym/tests/ops/test_*.py` — test specifications (20 files)
- `TileGym/src/tilegym/ops/ops.py` — operation interfaces

---

## Known Code Issues to Fix Before Eval

1. **`SkillEvolver.get_update_summary()` key mismatch:** Already fixed — reads both old (`num_skills_generated`) and new (`num_actions_generated`) history record keys.

2. **Correction prompt must avoid explicit-save keywords:** The LLM prompt for generating corrections explicitly forbids "always", "never", "from now on", "remember this" to ensure signals are classified as `implicit_correction` not `explicit_save`.

---

## Open Questions

1. **LLM cost:** Phase A: 31 teaching summaries + ~15 skill evolution calls. Phase B: up to 31 solve attempts + 31 comparisons + evolution calls per round. Phase C: 0 (just subprocess). Total: ~130 LLM calls per full run. Acceptable?

2. **Solve time for Phase C:** Each compute-eval round takes hours on B200. Multi-round Phase B is local (fast). Only Phase C needs the cluster.

3. **Which TileGym kernels to practice on:** All 31, or a curated subset that best covers the patterns needed for compute-eval problems?
