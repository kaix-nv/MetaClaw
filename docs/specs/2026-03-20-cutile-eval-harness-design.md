# cuTile Evaluation Harness for Conversation-Driven Skill Evolution

**Date:** 2026-03-20
**Status:** Draft
**Author:** Kaix + Claude

## Problem Statement

We built conversation-driven skill evolution into MetaClaw (skills extracted from user corrections, iterated via feedback). We need a concrete, measurable evaluation: does the system actually improve an agent's performance through conversation-driven skill learning?

cuTile kernel coding provides the test domain: 48 problems, Docker-based grading on B200 GPUs, clear pass/fail metrics. The existing `cutile-eval-kit` handles problem loading, agent execution, and evaluation.

## Design Goals

- Exercise the **full conversation pipeline**: `ConversationSignalDetector` → `SignalAggregator` → `SkillEvolver` → `SkillManager`
- Measure pass rate improvement across rounds of skill evolution
- Track per-problem status flips and attribute them to specific evolved skills
- Support configurable number of rounds (default: single round, option for multi-round convergence)
- Locally testable (skill evolution + reporting) without GPU/cluster
- Reuse existing eval kit infrastructure (shell wrappers, compute-eval API, Docker eval)

## Approach

Python orchestrator inside `MetaClaw/eval/cutile/` that:
1. Calls existing shell wrappers for B200 solve+eval (subprocess)
2. Parses eval failures and generates **realistic user correction messages** via LLM
3. Feeds corrections through MetaClaw's conversation signal detection pipeline
4. Evolved skills are injected into the next round's solve
5. Produces comparison reports with skill attribution

---

## Architecture Overview

```
MetaClaw/eval/cutile/
├── run_eval.py              # Main orchestrator (entry point)
├── failure_analyzer.py      # Find *-graded-solutions.jsonl, extract error context
├── correction_simulator.py  # LLM-generate natural corrections, feed through detector
├── skill_writer.py          # Write evolved skills to disk for next round
├── report.py                # Per-round and cross-round comparison reports
└── README.md                # Usage instructions
```

**Dependency direction:** `eval/cutile/` → MetaClaw components (SkillEvolver, SignalAggregator, etc.) → `cutile-eval-kit/scripts/` (shell wrappers via subprocess)

---

## Orchestration Loop

```
run_eval.py --rounds N --eval-kit-dir /path/to/cutile-eval-kit

Round 1 (baseline):
  1. run_solve_eval(skill_dir=None)
     → subprocess: compute-eval-run-cutile-opencode-gpt52.sh
     → subprocess: run_b200_local_compute_eval_evalonly.sh
     → produces *-graded-solutions.jsonl in eval-output/

  2. parse_failures()
     → finds *-graded-solutions.jsonl via glob (dynamic filename)
     → for each failed task: load problem-spec, error output, agent solution, test files

  3. simulate_corrections(failures)
     → for each failure: LLM generates natural correction message
     → feed each correction through ConversationSignalDetector
     → SignalAggregator accumulates signals

  4. evolve_skills()
     → SignalAggregator.should_evolve() → consume() → SkillEvolver.evolve()
     → returns SkillAction[] (create/update)

  5. write_skills(round=1)
     → SkillManager writes individual SKILL.md files to evolved-skills/round-1/
     → skill_writer.py concatenates base skill + all evolved skills into
       a single flat markdown file for injection (see Skill Injection section)

  6. generate_report(round=1)
     → per-task pass/fail/skip, active skills, evolved skills list

Round 2..N:
  Same, but --skill-dir points to evolved-skills/round-(N-1)/
  Report includes delta vs previous round

Final:
  Cross-round comparison report with skill attribution
```

---

## Correction Simulation

The core of the evaluation: turning automated failures into realistic user corrections that exercise the full conversation detection pipeline.

### Step 1: Extract failure context

For each failed task in eval-results.jsonl:

```python
failure = {
    "task_id": "cutile/3",
    "status": "failed",              # or "skipped"
    "error_message": "ImportError: cannot import name 'mma' from 'cuda.tile'",
    "problem_spec": load_problem_spec(task_id),
    "agent_solution": read_solution(task_id),
    "test_context": read_test_file(task_id),
}
```

### Step 2: LLM generates natural correction

```python
correction_prompt = f"""
You are a cuTile expert reviewing a student's failed solution.

Problem: {problem_spec.summary}
Student's code: {agent_solution[:500]}
Error: {error_msg}
Test expectation: {test_context[:300]}

Write a short, natural correction as if you were chatting with the student.
Start with what went wrong, then say what they should do instead.
Keep it under 3 sentences. Be specific about cuTile API usage.
"""

correction_message = call_llm(correction_prompt)
# e.g. "No, that's wrong. ct.mma doesn't exist — use ct.matmul instead,
#        and make sure your tile shapes match the tensor core requirements."
```

**Fallback:** If the LLM correction doesn't trigger detection (rare), prefix with a template: `"No, that's wrong. {llm_explanation}"` — guarantees the `ConversationSignalDetector` regex pattern matches.

### Step 3: Feed through real conversation pipeline

```python
turn_data = {
    "session_id": f"cutile-eval-round{round}-{task_id}",
    "turn_num": 1,
    "user_message": correction_message,
    "assistant_response": agent_solution,
    "active_skills": current_active_skills,
}
signals = detector.detect(turn_data)        # implicit_correction detected
aggregator.add(signals)
```

### Step 4: Evolve after all failures processed

```python
if aggregator.should_evolve():
    consumed = aggregator.consume()
    actions = await evolver.evolve(consumed, skill_manager.skills)
    for action in actions:
        if action.action == "create":
            skill_manager.add_skill(action.skill)
        elif action.action == "update":
            skill_manager.update_skill(action.skill["name"], action.skill)
```

### Signal confidence by error type

| Error Type | Confidence | Rationale |
|-----------|-----------|-----------|
| Compile/import error | 0.95 | Clear API misuse, easy to learn from |
| Wrong output | 0.80 | Logical error, may need pattern analysis |
| Timeout | 0.60 | Could be algorithmic or hardware-specific |
| Skip (env issue) | excluded | Not a solution quality problem |

Skipped tasks are excluded from correction simulation.

---

## Skill Attribution and Reporting

### Per-round data

```json
{
  "round": 1,
  "pass_rate": "20/48",
  "skills_injected": ["cutile-minimal-skill"],
  "skills_evolved": [],
  "per_task": {
    "cutile/0": {"status": "passed", "active_skills": ["cutile-minimal-skill"]},
    "cutile/1": {"status": "failed", "error_type": "compile", "error": "..."}
  }
}
```

### Cross-round comparison (final report)

```
=== cuTile Skill Evolution Report ===

Round  Pass  Fail  Skip  Skills Added
  1    20    25     3    (baseline)
  2    28    17     3    +use-ct-matmul-not-mma, +tile-shape-for-tensor-cores
  3    31    14     3    +reduction-pattern-with-ct-sum

Flips (fail→pass):
  Round 1→2: cutile/3, cutile/7, cutile/12, cutile/15, cutile/22, cutile/31
  Round 2→3: cutile/5, cutile/19, cutile/33

Regressions (pass→fail):
  (none)

Skill Attribution:
  use-ct-matmul-not-mma        → cutile/3, cutile/7, cutile/15 (3 tasks)
  tile-shape-for-tensor-cores   → cutile/12, cutile/22 (2 tasks)
  reduction-pattern-with-ct-sum → cutile/5, cutile/19, cutile/33 (3 tasks)
```

**Attribution logic:** A skill gets credit for a flip if it was added in round N and the task flipped fail→pass in round N. Correlation, not causation.

### Output files

```
results/
├── round-1/
│   ├── report.json              # structured per-task data
│   ├── eval-results.jsonl       # raw eval output
│   ├── corrections.jsonl        # simulated correction messages
│   └── evolved-skills/          # skills generated this round
├── round-2/
│   └── ...
└── summary.txt                  # human-readable cross-round comparison
```

---

## Configuration and CLI

```bash
python eval/cutile/run_eval.py \
  --rounds 3 \
  --eval-kit-dir /path/to/cutile-eval-kit \
  --results-dir eval/cutile/results \
  --skill-dir eval/cutile/evolved-skills \
  --model azure/openai/gpt-5.2 \
  --evolver-model gpt-5.2 \
  --max-attempts 3 \
  --task-ids ""
```

**Required env vars:** `AZURE_API_KEY` or `OPENAI_API_KEY`, B200 allocation (when on cluster).

**Config resolution:** CLI args > env vars > MetaClawConfig defaults.

### Local testing mode

```bash
python eval/cutile/run_eval.py \
  --local-test \
  --mock-eval-results path/to/previous/eval-results.jsonl
```

Skips actual solve+eval subprocess calls. Uses a previous run's eval-results.jsonl. Tests the full correction simulation → conversation detection → skill evolution → reporting pipeline without GPU/cluster.

---

## Integration with Existing Eval Kit

The orchestrator calls eval kit scripts via subprocess with env var overrides:

**For solve+eval:**
```bash
env MODEL=$model TASK_IDS=$task_ids SKILL_MD=$skill_path \
  bash $eval_kit_dir/scripts/compute-eval-run-cutile-opencode-gpt52.sh
```

**For eval-only rerun:**
```bash
env BASE_RUN=$run_root TASK_IDS=$task_ids \
  bash $eval_kit_dir/scripts/run_b200_local_compute_eval_evalonly.sh
```

The orchestrator reads outputs by searching for them (matching how the eval kit scripts find results):

**Eval results:** The eval-only script writes graded results to `$RUN_ROOT/eval-output/` with a dynamic filename pattern `*-graded-solutions.jsonl`. The orchestrator must find this file the same way the script does:
```python
import glob
graded = sorted(glob.glob(f"{run_root}/eval-output/*-graded-solutions.jsonl"))[-1]
```
Do NOT assume a fixed path like `$RUN_ROOT/eval-results.jsonl`.

**Agent solutions:** The generator packs solutions into a tar.gz datapack at `$OUTPUT_DIR/<release>-<model>-solutions.tar.gz`. To read individual solutions, use `compute_eval.data.data_pack.SolutionDatapack` to unpack, not raw filesystem paths. The per-task workspaces at `$RUN_ROOT/workspaces/<task_id>/` also contain the solutions before packing.

**Agent traces:** Available at `$RUN_ROOT/workspaces/<task_id>/` (the OpenCode working directory for each task).

---

## Skill Injection

The eval kit's generator (`compute-eval-opencode-cutile-gpt52.py`) accepts exactly one `--skill-md` argument — a single flat markdown file copied into each task workspace as `cutile-minimal-skill.md`. The OpenCode agent prompt references this file by name.

**Strategy: Concatenation.** `skill_writer.py` produces a single merged file per round:

```python
def write_merged_skill(base_skill_path, evolved_skills, output_path):
    """Concatenate base cuTile skill + all evolved skills into one file."""
    parts = [Path(base_skill_path).read_text()]
    parts.append("\n\n---\n\n# Evolved Skills (learned from previous failures)\n")
    for skill in evolved_skills:
        parts.append(f"\n## {skill['name']}\n_{skill['description']}_\n\n{skill['content']}\n")
    Path(output_path).write_text("\n".join(parts))
```

Output: `evolved-skills/round-N/cutile-skill-merged.md`

This file is passed to the shell wrapper as `SKILL_MD=.../cutile-skill-merged.md`. No changes to the generator script needed.

Individual SKILL.md files are also written (for MetaClaw's SkillManager tracking) but the merged file is what the agent sees.

---

## Correction Simulation: Avoiding Explicit Detection Mismatch

The `ConversationSignalDetector` skips implicit detection when an explicit pattern matches (e.g., "always use X" triggers `explicit_save` instead of `implicit_correction`). Since the correction simulator wants `implicit_correction` signals, the LLM prompt must avoid explicit-save trigger words.

**Constraint in the correction prompt:**

```
IMPORTANT: Do NOT use phrases like "always", "never", "from now on",
"remember this", or "save this as a skill". Start your correction with
"No," or "That's wrong," followed by what should be done differently.
```

**Fallback format:** If the LLM-generated correction still doesn't trigger implicit detection, force the format:
```python
if not detector.detect(turn_data):
    turn_data["user_message"] = f"No, that's wrong. {correction_message}"
    signals = detector.detect(turn_data)
```

---

## Known Code Issues to Fix Before Eval

1. **`SkillEvolver.get_update_summary()` key mismatch:** The refactored `evolve()` writes history records with keys `num_actions_generated` and `actions`, but `get_update_summary()` reads `num_skills_generated` and `skill_names`. Fix: update `get_update_summary()` to read the correct keys. (This affects reporting only, not the evolution pipeline.)

2. **`SkillEvolutionConfig` import path:** `SkillEvolutionConfig` lives in `metaclaw/signal_aggregator.py`, not `metaclaw/config.py`. The eval harness imports from `metaclaw.signal_aggregator`. The `MetaClawConfig.skill_evolution_config()` helper handles the conversion from flat config fields.

---

## Task ID Handling

When `--task-ids` is empty or omitted, all 48 tasks are run. The orchestrator passes this to subprocess as:

```python
env = {**os.environ, "TASK_IDS": task_ids if task_ids else ""}
subprocess.run([...], env=env)
```

An empty string `TASK_IDS=""` is the eval kit's convention for "all tasks". The orchestrator must never pass `None` — always use empty string for "all".

---

## Open Questions

1. **LLM cost:** Each failed task generates one LLM call for correction simulation, plus the SkillEvolver's LLM call for analysis. For 25 failures per round x 3 rounds = ~90 LLM calls total. Acceptable?

2. **Solve time:** Each round's solve+eval takes hours on B200. Multi-round (3+) means overnight runs. Is this expected?
