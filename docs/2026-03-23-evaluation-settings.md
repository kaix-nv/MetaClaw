# Evaluation Settings (v1 — cutile-eval-kit based)

**Date:** 2026-03-22
**Status:** Complete — results logged, switching to compute-eval-synth next

## Known Limitation

The `cutile-eval-kit` generator (`compute-eval-opencode-cutile-gpt52.py`) does NOT
isolate each problem in a Docker container. OpenCode runs with `--dir <workspace>`
but can escape and read other problem directories, host files, or previous solutions.
This is a potential information leakage confound.

**Next step:** Switch to `compute-eval-synth` (`generate_synth.py solve`) which runs
each problem in its own Docker container with isolated filesystem.

## Results (v1)

```
Method                     Pass   Fail   Skip   pass@1    Delta vs Baseline
──────────────────────────────────────────────────────────────────
Baseline (base skill only)  43      5      0    89.6%      —
Method 1 (direct extract)   44      4      0    91.7%    +1 task
Method 2 (multi-turn fix)   47      1      0    97.9%    +4 tasks
```

## Generation Settings

- **Generator:** `cutile-eval-kit/scripts/compute-eval-opencode-cutile-gpt52.py`
- **Agent:** OpenCode CLI (agentic, multi-turn, file read/write)
- **Model:** `aws/anthropic/bedrock-claude-opus-4-6` (Claude Opus 4.6)
- **API endpoint:** `https://inference-api.nvidia.com/v1`
- **Max attempts per problem:** 3
- **Problem set:** compute-eval release `2026-1`, group `cutile` (48 problems)
- **OpenCode provider:** Custom `nvidia` provider via `OPENCODE_CONFIG_CONTENT` JSON
- **OpenCode state:** Fresh XDG dirs per run to avoid stale session DB

## Evaluation Settings

- **Evaluator:** `compute_eval.evaluation.evaluate_functional_correctness`
- **compute-eval commit:** `e01a5d2` ("Release 2026.1 (#11)")
- **Docker image:** `local/compute-eval-python:13.1.0` (built from `docker/Dockerfile.python-cuda13`)
- **Docker backend:** CLI (`COMPUTE_EVAL_DOCKER_BACKEND=cli`) — SDK fails on B200
- **docker_evaluator.py:** Patched version from seedbot workspace with CLI backend support
- **Image registry:** `local` (avoids NGC dependency)
- **Eval mode:** Docker
- **k:** 1
- **n_workers:** 1
- **GPU:** NVIDIA B200 (SM 10.0) on Computelab

## Skill Settings

### Baseline
- `cutile-minimal-skill.md` (~5900 lines)
- Built from NVIDIA official cuTile Python docs via `build_cutile_minimal_skill.py`
- Contains: all ct.* API names, overview pages, API docs for 33 symbols

### Method 1 (Direct Extraction)
- Phase A only: LLM reads 22 TileGym kernels → teaching summaries → SkillEvolver
- Evolved skills: `cutile-kernel-patterns`, `cutile-attention-kernels`
- Merged file: base skill + 2 evolved skills

### Method 2 (Multi-Turn Evolution)
- Phase A: same as Method 1
- Phase B: agent attempts kernels, sees reference on failure, re-attempts tested on GPU
  - Docker image: `local/compute-eval-tilegym:13.1.0` (extends compute-eval with TileGym)
  - 6 kernels fixed: dropout, group_gemm, mla_decoding, silu_and_mul, splitk_reduce, swiglu
  - 3 evolved skills: `cutile-kernel-patterns`, `cutile-attention-kernels`, `cutile-elementwise-and-1d-kernels`
- Merged file: base skill + 3 evolved skills (from Phase A + Phase B)

## Run Directories

```
runs/cutile-eval-control-20260322-033800/   # Baseline (43/48)
runs/cutile-eval-method1-20260322-033320/   # Method 1 (44/48)
runs/cutile-eval-method2-20260322-210822/   # Method 2 (47/48)
```

## Run Scripts

```
runs/run-control.sh          # Baseline: OpenCode + base skill
runs/run-method1.sh          # Method 1: OpenCode + extracted skills
runs/run-method2-full.sh     # Method 2: Phase A+B + OpenCode + eval (all-in-one)
runs/eval-all.sh             # Evaluation only (Docker grading)
runs/eval_patched.py         # CLI Docker backend wrapper
runs/test-docker-tilegym.sh  # TileGym Docker validation test
```

## Paths

```
compute-eval/                              # github.com/NVIDIA/compute-eval @ e01a5d2
TileGym/                                   # TileGym kernel library
cutile-eval-kit/                           # OpenCode-based generator scripts
eval/cutile/cutile-minimal-skill.md        # Base skill (NVIDIA docs)
eval/cutile/results/method-1-direct/       # Method 1 skills + merged
eval/cutile/results/method-2-multiturn/    # Method 2 skills + merged
eval/cutile/Dockerfile.tilegym             # TileGym Docker image
```
