# cuTile Evaluation Harness: Direct Extraction vs Multi-Turn Skill Evolution

**Date:** 2026-03-20
**Status:** Draft (v3 — two-method comparison)
**Author:** Kaix + Claude

## Problem Statement

We built conversation-driven skill evolution into MetaClaw. We need to evaluate whether it works — and specifically whether the multi-turn feedback loop adds value over simple one-shot skill extraction.

cuTile kernel coding provides the test domain. **TileGym** (31 cuTile kernel implementations) is the skill source. **compute-eval** (48 cuTile problems) is the held-out test.

## Research Question

Does iterative, conversation-driven skill evolution outperform one-shot skill extraction for improving an LLM agent's cuTile kernel coding ability — without any model training?

## Experimental Design

Three conditions, same compute-eval benchmark:

```
Method                    Skill Source    Feedback Loop?   Expected
──────────────────────────────────────────────────────────────────
No skills (control)       None            No               X/48
Method 1: Direct extract  TileGym         No               Y/48
Method 2: Multi-turn      TileGym         Yes              Z/48
```

Both methods are training-free (LLM API calls only). The comparison isolates the value of the conversation-driven feedback loop.

---

## Method 1: Direct Extraction (Baseline)

One-shot skill generation. LLM reads TileGym kernels, produces skills, done.

### Process

```
For each of 31 TileGym kernels:
  1. Read kernel implementation + test file
  2. LLM generates a teaching summary (one call per kernel)
  3. Feed through ConversationSignalDetector → explicit_save
  4. SignalAggregator triggers → SkillEvolver synthesizes skills

Output: skill bank (written to disk)
No re-attempts, no corrections, no iteration.
```

### Teaching summary generation

```python
teaching_prompt = f"""
You are a cuTile expert writing a teaching note for a student.

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
patterns and API usage. Start with "Remember this:" to make it clear
this is a skill to save. Be specific about ct.* API calls, tile sizes,
and hardware features (TMA, tensor cores).
"""
```

### Signal flow

```python
turn_data = {
    "session_id": f"direct-extract-{kernel_name}",
    "turn_num": 1,
    "user_message": teaching_message,      # "Remember this: ..." → explicit_save
    "assistant_response": kernel_source,
    "active_skills": [],
}
signals = detector.detect(turn_data)
aggregator.add(signals)
```

After all 31 kernels, SkillEvolver synthesizes across all signals → **~10-20 skills**.

### Cost

- 31 LLM calls (teaching summaries) + 1 evolution call
- ~5 minutes, no GPU

---

## Method 2: Multi-Turn Skill Evolution (Proposed)

Iterative feedback loop. Agent attempts kernels, gets corrected, skills evolve, agent re-attempts with evolved skills.

### Process

```
Phase A (Study): Same as Method 1 → initial skill bank

Phase B (Practice), per round:
  For each of 31 TileGym kernels:
    1. Agent attempts implementation given only the test file + current skills
    2. Compare attempt vs reference implementation
    3. If wrong: LLM generates correction → ConversationSignalDetector → implicit_correction
    4. After all kernels: corrections clustered → SkillEvolver evolves
    5. Re-attempt failed kernels with evolved skills (next round)

  Stop when: no new corrections, or --rounds limit

Phase C (Evaluate): Same benchmark for both methods
```

### Phase B detail: Solve-and-check

**Given to agent:** test file + current skills
**NOT given:** reference implementation

```python
solve_prompt = f"""
You are solving a cuTile kernel implementation problem.

Test specification:
```python
{test_source}
```

{current_skills_text}

Write the cuTile kernel implementation that passes these tests.
"""

agent_solution = call_llm(solve_prompt)
```

### Phase B detail: Correction generation

```python
comparison_prompt = f"""
Compare this student's cuTile kernel with the reference.

Student attempt:
```python
{agent_solution[:1000]}
```

Reference (correct):
```python
{reference[:1000]}
```

If wrong, write a correction (2-3 sentences). Start with "No, that's wrong."
Be specific about cuTile API usage.
Do NOT use "always", "never", "from now on", or "remember this".

If correct, respond with just "CORRECT".
"""
```

### Phase B detail: Correction clustering

Before feeding to SkillEvolver, group corrections by error pattern (simple keyword matching):

```
20 corrections → 4-5 clusters:
  Cluster A: "used manual loops instead of ct.* builtins" (6)
  Cluster B: "wrong tile shapes for tensor cores" (4)
  Cluster C: "missing TMA configuration" (3)
  Cluster D: "incorrect memory layout / transpose" (3)
  Cluster E: "miscellaneous API misuse" (4)
```

### Phase B detail: Contrastive diagnosis

For each cluster, the SkillEvolver LLM sees:
1. Agent's wrong attempts
2. Corrections
3. **Which skills were active when these failures happened**

This enables three distinct diagnoses:
- Skills **MISSING** → create new skill
- Skill **PRESENT but WRONG** → update that skill
- Skill **PRESENT but IGNORED** → make skill more specific/forceful

### Phase B detail: Conversational evolution

The evolution itself is a multi-turn conversation (not a single LLM call):

```
Turn 1 (system): "Your agent failed 6 kernels with manual loops.
                  Skills active: [prefer-builtin-ct-ops]. Corrections: [...]"

Turn 2 (evolver): "I'll split prefer-builtin-ct-ops into two specific skills:
                   use-ct-matmul-for-gemm and use-ct-sum-for-reduction"

Turn 3 (system): [re-attempt results] "Agent now passes 4/6. Still fails
                  2 — uses ct.sum but wrong axis."

Turn 4 (evolver): "Updated use-ct-sum-for-reduction with axis guidance."
```

Each round of Phase B's re-attempt loop is also a turn in the evolution conversation. The evolver learns from seeing its own skills succeed or fail.

### Phase B detail: Reflection cycle

After proposing skills, one reflection round (from MemSkill's Designer):

```
"You proposed 'prefer-builtin-ct-ops'. Is this specific enough?
 Would the agent know WHICH builtins for WHICH patterns?"
→ LLM refines, possibly splits into parent + children skills
```

### Phase B detail: Re-attempt and second-order corrections

Agent re-attempts only previously failed kernels with new skills:

| Outcome | Meaning | Evolution action |
|---------|---------|-----------------|
| Now passes | Skill worked | Boost confidence |
| Same error | Skill ineffective | Correction includes "you have skill X but still failed" → update |
| Different error | Partially helped | New correction → may need additional skill |

"Still fails with same error" produces **second-order corrections** — the system learns from its own skill evolution failures.

### Phase B detail: Convergence

Stop when:
- All kernels pass (converged)
- Same corrections repeat for 2 consecutive rounds (plateaued)
- `--rounds` limit reached

### Cost

- Phase A: same as Method 1 (~32 LLM calls)
- Phase B per round: up to 31 solve + 31 compare + clustering + evolution conversation (~70 LLM calls)
- 2-3 rounds typical → ~180-250 LLM calls total
- ~30 minutes, no GPU

---

## Evaluation (Phase C) — Same for Both Methods

### Skill injection

`skill_writer.py` concatenates base cuTile skill + all evolved skills into one flat markdown file:

```python
def write_merged_skill(base_skill_path, evolved_skills, output_path):
    parts = [Path(base_skill_path).read_text()]
    parts.append("\n\n---\n\n# Evolved Skills (learned from TileGym)\n")
    for skill in evolved_skills:
        parts.append(f"\n## {skill['name']}\n_{skill['description']}_\n\n{skill['content']}\n")
    Path(output_path).write_text("\n".join(parts))
```

Passed as `SKILL_MD=.../cutile-skill-merged.md` to the eval kit's generator script.

### Benchmark execution

```bash
# Control (no skills):
env MODEL=$model SKILL_MD="" bash $eval_kit_dir/scripts/compute-eval-run-cutile-opencode-gpt52.sh

# Method 1 (direct extraction):
env MODEL=$model SKILL_MD=$method1_skills bash $eval_kit_dir/scripts/...

# Method 2 (multi-turn):
env MODEL=$model SKILL_MD=$method2_skills bash $eval_kit_dir/scripts/...
```

### Reading results

Eval script writes `*-graded-solutions.jsonl` to `$RUN_ROOT/eval-output/`:
```python
graded = sorted(glob.glob(f"{run_root}/eval-output/*-graded-solutions.jsonl"))[-1]
```

---

## Reporting

### Primary result table

```
Method                    Pass   Fail   Skip   Delta vs Control
──────────────────────────────────────────────────────────────
No skills (control)       20     25      3     —
Method 1: Direct extract  Y      ...     3     +Y-20
Method 2: Multi-turn      Z      ...     3     +Z-20
```

### Per-problem flip analysis

```
Flips (fail→pass):
  Control → Method 1: cutile/3, cutile/7, cutile/12
  Control → Method 2: cutile/3, cutile/7, cutile/12, cutile/5, cutile/15, cutile/40
  Method 1 only: (none — subset of Method 2)
  Method 2 only: cutile/5, cutile/15, cutile/40 (these required iterative refinement)

Regressions (pass→fail):
  (ideally none)
```

### Skill attribution

```
Skills from Method 1 (direct extraction):
  softmax-tma-loading          → cutile/3, cutile/12
  prefer-builtin-ct-ops        → cutile/7

Additional skills from Method 2 (multi-turn):
  use-ct-sum-for-reduction     → cutile/5, cutile/40
  tile-shape-tensor-cores      → cutile/15
  (these skills required correction feedback to generate —
   direct extraction missed them)
```

### Phase B evolution trace

```
Round 1: 20 corrections → 5 new skills → 12 kernels still failing
Round 2: 12 corrections → 2 updates + 1 new → 5 kernels still failing
Round 3: 5 corrections → 1 update → 3 kernels still failing (plateau)
```

### Output files

```
results/
├── method-1-direct/
│   ├── teaching-summaries.jsonl
│   ├── evolved-skills/
│   └── compute-eval-results.jsonl
├── method-2-multiturn/
│   ├── phase-a/
│   │   ├── teaching-summaries.jsonl
│   │   └── evolved-skills/
│   ├── phase-b/
│   │   ├── round-1/
│   │   │   ├── attempts.jsonl
│   │   │   ├── corrections.jsonl
│   │   │   ├── evolution-conversation.jsonl
│   │   │   └── evolved-skills/
│   │   └── round-2/ ...
│   └── compute-eval-results.jsonl
├── control/
│   └── compute-eval-results.jsonl
├── comparison.json
└── summary.txt
```

---

## Architecture

```
MetaClaw/eval/cutile/
├── run_eval.py              # Main orchestrator
├── phase_a_study.py         # Read TileGym → teaching summaries → skills
├── phase_b_practice.py      # Solve-and-check → corrections → evolve → re-attempt
├── phase_c_evaluate.py      # Run compute-eval benchmark
├── failure_analyzer.py      # Parse graded-solutions.jsonl
├── correction_simulator.py  # Generate corrections from reference comparison
├── correction_clusterer.py  # Group corrections by error pattern
├── skill_writer.py          # Write/merge skills for injection
├── report.py                # Comparison reports
└── README.md
```

---

## CLI

```bash
# Full comparison (all three conditions):
python eval/cutile/run_eval.py \
  --tilegym-dir /path/to/TileGym \
  --eval-kit-dir /path/to/cutile-eval-kit \
  --results-dir eval/cutile/results \
  --model azure/openai/gpt-5.2 \
  --evolver-model gpt-5.2 \
  --base-skill /path/to/cutile-minimal-skill.md \
  --rounds 3

# Method 1 only (local, no GPU):
python eval/cutile/run_eval.py --method direct --tilegym-dir ...

# Method 2 only (local, no GPU for Phase A+B):
python eval/cutile/run_eval.py --method multiturn --tilegym-dir ... --rounds 3

# Phase C only (needs B200):
python eval/cutile/run_eval.py --method evaluate \
  --skills-from results/method-2-multiturn/phase-b/round-3/evolved-skills/ \
  --eval-kit-dir ...

# Local test (skip Phase C, use mock results):
python eval/cutile/run_eval.py --local-test \
  --mock-eval-results path/to/previous/eval-results.jsonl
```

---

## Integration

### MetaClaw components

All from `metaclaw.*`:
- `ConversationSignalDetector` — explicit_save (Phase A), implicit_correction (Phase B)
- `SignalAggregator` + `SkillEvolutionConfig` (from `metaclaw.signal_aggregator`)
- `SkillEvolver` — skill synthesis, conversational evolution in Method 2
- `SkillManager` — skill storage, confidence tracking, hierarchy
- `SkillAction` — create/update/deprecate/link

### Eval kit scripts (Phase C only)

Via subprocess:
- `cutile-eval-kit/scripts/compute-eval-run-cutile-opencode-gpt52.sh`
- `cutile-eval-kit/scripts/run_b200_local_compute_eval_evalonly.sh`

### TileGym (read-only)

- `TileGym/src/tilegym/ops/cutile/*.py` — 31 kernel implementations
- `TileGym/tests/ops/test_*.py` — 20 test files
- `TileGym/src/tilegym/ops/ops.py` — operation interfaces

---

## Correction Simulation Constraints

The `ConversationSignalDetector` skips implicit detection when an explicit pattern matches. Correction prompts must avoid explicit-save trigger words ("always", "never", "from now on", "remember this").

Fallback if correction doesn't trigger detection:
```python
if not detector.detect(turn_data):
    turn_data["user_message"] = f"No, that's wrong. {correction_message}"
```

---

## Task ID Handling

`--task-ids ""` (empty string) means all tasks. The orchestrator passes `TASK_IDS=""` to subprocess. Never pass `None`.

---

## Open Questions

1. **LLM cost acceptable?** Method 1: ~32 calls. Method 2: ~250 calls. Phase C: 0 (subprocess).
2. **Which TileGym kernels?** All 31, or a curated subset covering compute-eval patterns?
3. **Multiple eval runs for statistical significance?** Each compute-eval run has some variance. Run 3x and report mean ± std?
