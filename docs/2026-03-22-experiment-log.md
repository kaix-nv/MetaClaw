# Experiment Log: Conversation-Driven Skill Evolution for cuTile

**Date:** 2026-03-20 to 2026-03-22
**Author:** Kaix + Claude
**Branch:** `kaix/conversation-driven-skill-evolution`

## Goal

Test whether conversation-driven skill evolution can improve an LLM agent's cuTile kernel coding ability on the compute-eval benchmark, without any model training.

## Final Results

```
Method                     Pass   Fail   pass@1    Delta vs Baseline
──────────────────────────────────────────────────────────────────
Baseline (base skill only)  43      5    89.6%      —
Method 1 (direct extract)   44      4    91.7%    +1 task
Method 2 (multi-turn fix)   47      1    97.9%    +4 tasks
```

## System Architecture

### Skill Evolution Pipeline (MetaClaw)

Built a pluggable signal-based skill evolution system:
- `ConversationSignalDetector` — detects explicit ("remember this") and implicit ("no, that's wrong") signals
- `SignalAggregator` — buffers signals, triggers evolution at thresholds
- `SkillEvolver` — LLM analyzes signals, proposes skill CRUD actions
- `SkillManager` — stores skills as markdown files with confidence tracking and hierarchy

### Evaluation Pipeline

- **Skill source:** TileGym (22 cuTile kernel implementations + tests)
- **Benchmark:** compute-eval (48 cuTile problems, Docker-graded on B200)
- **Agent:** OpenCode CLI (agentic, multi-turn, reads files and writes solutions)

## Method 1: Direct Extraction (Phase A only)

LLM reads each TileGym kernel → generates teaching summary → feeds through conversation detector → SkillEvolver synthesizes skills.

- 22 kernels → 22 signals → 2-3 evolved skills
- ~5 minutes, ~23 LLM calls
- No GPU needed

## Method 2: Multi-Turn Evolution (Phase A + Phase B)

Phase A: Same as Method 1 (direct extraction)

Phase B: Agent attempts to implement each kernel from test spec alone. When it fails, it sees the reference implementation, re-attempts, and if the re-attempt passes the real TileGym test suite (in Docker on B200), a skill is extracted from the diff between the failed and fixed versions.

- Round 1: 6/22 kernels fixed → 3 new skills extracted
- Round 2: 0/16 remaining fixed → plateau, stopped
- ~2 hours on B200, ~200 LLM calls

### Phase B Fixed Kernels
`dropout`, `group_gemm`, `mla_decoding`, `silu_and_mul`, `splitk_reduce`, `swiglu`

## Key Debugging Journey

### 1. OpenCode CLI hanging on B200 (Session DB)

**Problem:** OpenCode CLI timed out after 600s on every attempt, producing zero files.

**Root cause:** Stale SQLite session database from previous runs. OpenCode stores session state in `~/.local/share/opencode/` (or similar XDG path). On compute nodes, this directory persisted across jobs.

**Fix:** Set fresh `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `XDG_STATE_HOME` per run.

**Time spent:** ~4 hours debugging.

### 2. OpenCode provider configuration

**Problem:** Model name `aws/anthropic/bedrock-claude-opus-4-6` not recognized by OpenCode. Error: `ProviderModelNotFoundError`.

**Root cause:** OpenCode's `-m` flag expects a provider-prefixed model name, and the provider must be registered in config. The NVIDIA inference API isn't a built-in provider.

**Fix:** Set `OPENCODE_CONFIG_CONTENT` env var with a JSON config that registers a `nvidia` provider with `@ai-sdk/openai-compatible` SDK, mapping the model to `https://inference-api.nvidia.com/v1`.

### 3. Docker SDK failing on B200 (RemoteDisconnected)

**Problem:** compute-eval's Docker evaluator crashed with `ConnectionError: RemoteDisconnected` when creating containers.

**Root cause:** Docker Python SDK (`docker-py`) has a known issue on B200 compute nodes — the Docker daemon drops SDK connections for container creation.

**Fix:** Copied the CLI-backend-patched `docker_evaluator.py` from the seedbot workspace, which uses `subprocess.run(["docker", ...])` instead of the SDK. Activated via `COMPUTE_EVAL_DOCKER_BACKEND=cli`.

### 4. compute-eval temp directory caching

**Problem:** Method 1 and Method 2 generations shared the same temp directory (named after model), causing "Skipping, already have solutions" for all tasks.

**Root cause:** `generate_samples` caches per-task JSONL files in a `temp_dir` defaulting to the model name. All three conditions used the same model.

**Fix:** Added `--temp_dir="$OUT_DIR/temp"` to each generation script.

### 5. Skill injection: system_prompt vs workspace file

**Problem:** Initially used `--system_prompt` to inject skills into compute-eval's `generate_samples`. This replaced the default CUDA engineer prompt entirely.

**Discovery:** The cutile-eval-kit uses OpenCode, which reads skills as a file in the workspace (`cutile-minimal-skill.md`). This is a fundamentally different and better approach — the agent reads the skill alongside the problem spec and tests.

**Fix:** Switched to OpenCode-based generation with skills passed via `--skill-md`.

### 6. Phase B: Learning from failures vs successes

**Problem (v1):** Phase B generated skills from correction messages about failures ("don't do X"). These produced overly restrictive rules like "Do NOT compute grid dims in Python host code" that caused regressions (Method 2 scored 42/48, worse than baseline's 43/48).

**Problem (v2):** Changed to only learn from successful fixes, but the agent couldn't fix any kernel from a text hint alone (0/22 fixed).

**Problem (v3):** Showed agent the full reference code, but LLM-as-judge comparison was too strict — truncated large kernels to 1500 chars, rejected functionally equivalent code.

**Final fix:**
- Show reference code in re-attempt prompt (up to 6000 chars)
- Use real GPU tests (TileGym pytest in Docker) instead of LLM comparison
- Only extract skills from verified successful fixes
- Strip markdown fences from LLM output before writing to .py files

### 7. TileGym test execution in Docker

**Problem (v1):** Ran pytest directly on B200 host. Failed because `pip install` on shared nodes is forbidden, and cuTile cache filled the home directory.

**Problem (v2):** Mounted TileGym into the compute-eval Docker image. Failed because `pip install -e .` needed writable directory, and `pip` wasn't in the image.

**Problem (v3):** Tried `PYTHONPATH` hack. Failed with import errors.

**Final fix:** Built a custom Docker image `local/compute-eval-tilegym:13.1.0` that extends `compute-eval-python:13.1.0` with TileGym pre-installed (including `git` for the `cuda-tile-experimental` dependency). `KernelTester` mounts only the single modified kernel file into the container.

### 8. Markdown fences in LLM output

**Problem:** All kernels got pytest exit code 2 (collection error) even with valid reference code visible.

**Root cause:** LLM wraps code in ` ```python ... ``` ` fences. The `KernelTester` wrote this raw to the `.py` file, causing Python syntax errors.

**Fix:** Added `_strip_markdown_fences()` to extract clean Python from LLM output.

## Infrastructure Setup

### Docker Images

| Image | Purpose | Build |
|-------|---------|-------|
| `local/compute-eval-python:13.1.0` | compute-eval benchmark grading | `docker build -f compute-eval/docker/Dockerfile.python-cuda13` |
| `local/compute-eval-tilegym:13.1.0` | Phase B kernel testing | `docker build -f eval/cutile/Dockerfile.tilegym` |

Both saved as tars for B200 node loading.

### B200 Job Submission

```bash
# Generation (OpenCode + skill injection)
crun -q 'gpu.product_name=*B200*' --gpus 1 -t 4:00:00 -b runs/run-control.sh
crun -q 'gpu.product_name=*B200*' --gpus 1 -t 4:00:00 -b runs/run-method1.sh

# Method 2 full pipeline (Phase A + B + generation + eval)
crun -q 'gpu.product_name=*B200*' --gpus 1 -t 8:00:00 -b runs/run-method2-full.sh

# Evaluation only
crun -q 'gpu.product_name=*B200*' --gpus 1 -t 2:00:00 -b runs/eval-all.sh
```

### LLM API

- Endpoint: `https://inference-api.nvidia.com/v1`
- Model: `aws/anthropic/bedrock-claude-opus-4-6` (Claude Opus 4.6)
- API key: `$API_KEY` env var

## File Structure

```
MetaClaw/
├── metaclaw/                    # Core skill evolution pipeline
│   ├── skill_signal.py          # SkillSignal, SkillAction dataclasses
│   ├── signal_aggregator.py     # Buffer + trigger logic
│   ├── conversation_signal_detector.py  # Explicit/implicit detection
│   ├── skill_evolver.py         # LLM-based skill synthesis
│   ├── skill_manager.py         # Skill CRUD + hierarchy + confidence
│   ├── feedback_collector.py    # REST endpoints for feedback
│   ├── prm_signal_adapter.py    # PRM backward compat
│   ├── api_server.py            # Signal hooks + feedback endpoints
│   └── trainer.py               # Signal aggregator wiring
├── eval/cutile/                 # cuTile evaluation harness
│   ├── run_eval.py              # Main orchestrator
│   ├── phase_a_study.py         # Direct extraction
│   ├── phase_b_practice.py      # Multi-turn with GPU tests
│   ├── kernel_tester.py         # Docker-based pytest runner
│   ├── correction_simulator.py  # Reference comparison
│   ├── skill_writer.py          # Merge skills for injection
│   ├── report.py                # Comparison reports
│   ├── llm_client.py            # NVIDIA API client
│   ├── tilegym_loader.py        # Load kernels + tests
│   └── Dockerfile.tilegym       # TileGym Docker image
├── runs/                        # Execution scripts + results
│   ├── run-control.sh           # Baseline generation
│   ├── run-method1.sh           # Method 1 generation
│   ├── run-method2-full.sh      # Method 2 full pipeline
│   ├── eval-all.sh              # Docker-based evaluation
│   └── eval_patched.py          # CLI Docker backend for compute-eval
├── compute-eval/                # NVIDIA compute-eval (cloned)
├── TileGym/                     # TileGym kernel library (cloned)
└── cutile-eval-kit/             # cuTile eval kit (reference)
```

## Lessons Learned

1. **Learn from successes, not failures.** Skills extracted from failure corrections produce restrictive rules that cause regressions. Skills from successful fixes capture positive, working patterns.

2. **Real tests beat LLM-as-judge.** LLM comparison is unreliable for code correctness — it truncates, misjudges, and can't handle functional equivalence. Running actual tests is the ground truth.

3. **Docker everything.** Never install packages on shared compute nodes. Bake dependencies into Docker images and mount only what changes.

4. **Strip LLM formatting.** LLM output always includes markdown fences. Any pipeline that writes LLM output to executable files must strip formatting first.

5. **Fresh state per run.** Agents (OpenCode) and tools with persistent state (session DBs, temp caches) need clean directories per run to avoid corruption.

6. **The skill injection interface matters.** System prompt replacement is crude — workspace file injection (OpenCode reads skills as context) produces better results because the agent can selectively apply relevant patterns.
