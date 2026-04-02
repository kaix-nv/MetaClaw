# Autoresearch-Style cuTile Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a program.md-driven cuTile solver that uses autoresearch's pattern (modify → test → keep/discard → repeat) to solve TileGym kernels and extract skills from debug journeys.

**Architecture:** `program.md` instructs the agent. `autoresearch_solver.py` sets up workspaces and launches OpenCode in Docker per kernel. `git_trace_parser.py` extracts error→fix pairs from git history. Existing `skill_synthesizer.py` and MetaClaw pipeline handle skill evolution.

**Tech Stack:** Python 3.10+, Docker, OpenCode CLI, git, MetaClaw components (existing).

**Spec:** `docs/specs/2026-04-02-autoresearch-cutile-solver.md`

---

## File Structure

```
eval/cutile/
├── program.md                   [NEW]  Agent instruction document
├── autoresearch_solver.py       [NEW]  Workspace setup + Docker launch orchestrator
├── git_trace_parser.py          [NEW]  Parse git diffs + results.log → error→fix pairs
├── run_skill_synthesis.py       [MODIFY]  Updated to use autoresearch solver
├── skill_synthesizer.py         [EXISTING]  Reuse as-is
├── tilegym_loader.py            [EXISTING]  Reuse as-is
├── llm_client.py                [EXISTING]  Reuse as-is
├── skill_writer.py              [EXISTING]  Reuse as-is

tests/
├── test_git_trace_parser.py     [NEW]
```

---

### Task 1: program.md

**Files:**
- Create: `eval/cutile/program.md`

- [ ] **Step 1: Write program.md**

```markdown
# cuTile Kernel Implementation

## Goal
Implement a cuTile kernel that passes the test in `test/`.

## Setup
1. Read the cuTile skill in `.opencode/skill/cutile-python/SKILL.md` to understand the cuTile API
2. Read `problem-spec.yaml` to understand what kernel to implement
3. Read the test file in `test/` to understand expected inputs, outputs, and function signatures
4. Initialize git tracking: `git init && git add -A && git commit -m "initial workspace"`

## Experiment Loop
Repeat until the test passes or you have made 15 attempts:

1. **Write or modify** `solution.py` with your cuTile kernel implementation
2. **Commit your attempt**: `git add solution.py && git commit -m "attempt N: <brief description of change>"`
3. **Run the test**:
   ```
   cd /testbed && PYTHONPATH=/testbed python -m pytest test/ -x -v 2>&1 | tee -a results.log
   ```
4. **Check the result**:
   - If all tests **PASSED**: append `FINAL: PASS attempts=N` to results.log, then stop
   - If tests **FAILED**: read the error message carefully, understand what went wrong, then go back to step 1

## Rules
- Only create or modify `solution.py` — never modify files in `test/`
- Use the cuTile API: `import cuda.tile as ct`
- Every solution must include `@ct.kernel` decorator and `ct.launch()` call
- Use `ct.float32` accumulators for numerical stability in reductions and matmul
- Always commit before running the test so we can track every iteration
- Read error messages carefully — they tell you exactly what went wrong
- If you get a compilation error, check your ct.* API usage against the skill doc
- If you get a numerical mismatch, check accumulator dtype and reduction order

## When Done
Append your final status to results.log:
```
FINAL: PASS attempts=<N> kernel=<kernel_name>
```
or if you could not solve it after 15 attempts:
```
FINAL: FAIL attempts=15 kernel=<kernel_name>
```
```

- [ ] **Step 2: Commit**

```bash
cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw
git add eval/cutile/program.md
git commit -m "feat(eval): add program.md for autoresearch-style cuTile solving

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Git Trace Parser

**Files:**
- Create: `eval/cutile/git_trace_parser.py`
- Test: `tests/test_git_trace_parser.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_git_trace_parser.py
import os
import subprocess
import pytest
from pathlib import Path
from eval.cutile.git_trace_parser import GitTraceParser


@pytest.fixture
def git_workspace(tmp_path):
    """Create a fake git workspace simulating an agent's solve attempts."""
    ws = tmp_path / "workspace"
    ws.mkdir()

    # Init git
    subprocess.run(["git", "init"], cwd=ws, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=ws, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=ws, capture_output=True)

    # Initial commit
    (ws / "problem-spec.yaml").write_text("kernel: softmax\n")
    subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial workspace"], cwd=ws, capture_output=True)

    # Attempt 1: bad code
    (ws / "solution.py").write_text("import cuda.tile as ct\n\ndef bad():\n    pass\n")
    subprocess.run(["git", "add", "solution.py"], cwd=ws, capture_output=True)
    subprocess.run(["git", "commit", "-m", "attempt 1: first try"], cwd=ws, capture_output=True)

    # Attempt 2: fixed code
    (ws / "solution.py").write_text("import cuda.tile as ct\n\n@ct.kernel\ndef good():\n    ct.load()\n")
    subprocess.run(["git", "add", "solution.py"], cwd=ws, capture_output=True)
    subprocess.run(["git", "commit", "-m", "attempt 2: fix kernel decorator"], cwd=ws, capture_output=True)

    # Write results.log with error from attempt 1
    (ws / "results.log").write_text(
        "FAILED: RuntimeError: ct.mma expects 2D tiles\n"
        "PASSED\n1 passed in 5.2s\n"
        "FINAL: PASS attempts=2 kernel=softmax\n"
    )

    return ws


def test_parse_git_history(git_workspace):
    parser = GitTraceParser()
    pairs = parser.parse(str(git_workspace))
    assert len(pairs) >= 1
    pair = pairs[0]
    assert "code_before" in pair
    assert "code_after" in pair
    assert "bad()" in pair["code_before"]
    assert "good()" in pair["code_after"]


def test_parse_extracts_errors(git_workspace):
    parser = GitTraceParser()
    pairs = parser.parse(str(git_workspace))
    assert pairs[0].get("error_message", "")
    assert "ct.mma" in pairs[0]["error_message"] or len(pairs[0]["error_message"]) > 0


def test_parse_detects_pass(git_workspace):
    parser = GitTraceParser()
    result = parser.get_final_status(str(git_workspace))
    assert result["passed"] is True
    assert result["attempts"] == 2


def test_parse_empty_workspace(tmp_path):
    parser = GitTraceParser()
    pairs = parser.parse(str(tmp_path))
    assert pairs == []


def test_parse_no_solution(tmp_path):
    """Workspace with git but no solution.py commits."""
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init"], cwd=ws, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=ws, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=ws, capture_output=True)
    (ws / "readme").write_text("hi")
    subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=ws, capture_output=True)
    parser = GitTraceParser()
    pairs = parser.parse(str(ws))
    assert pairs == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_git_trace_parser.py -v`

- [ ] **Step 3: Implement git_trace_parser.py**

```python
# eval/cutile/git_trace_parser.py
"""Parse git history from autoresearch-style solve workspaces into error→fix pairs."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class GitTraceParser:
    """Extract error→fix pairs from git commit history of a solve workspace."""

    def parse(self, workspace_dir: str) -> list[dict]:
        """Parse git history of a workspace to find error→fix pairs.

        Looks for consecutive commits that modify solution.py.
        Pairs each commit with errors from results.log between those commits.
        """
        ws = Path(workspace_dir)
        if not (ws / ".git").exists():
            return []

        # Get list of commits that touch solution.py
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--follow", "--", "solution.py"],
                cwd=ws, capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError:
            return []

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return []

        # Commits are newest-first, reverse for chronological order
        commits = [l.split()[0] for l in reversed(lines)]

        # Read errors from results.log
        errors = self._parse_results_log(ws / "results.log")

        # Extract pairs from consecutive commits
        pairs = []
        for i in range(len(commits) - 1):
            old_sha = commits[i]
            new_sha = commits[i + 1]

            code_before = self._git_show(ws, old_sha, "solution.py")
            code_after = self._git_show(ws, new_sha, "solution.py")

            if not code_before or not code_after:
                continue

            # Find error message between these attempts
            error_msg = errors[i] if i < len(errors) else ""

            # Get commit message for context
            commit_msg = self._git_commit_msg(ws, new_sha)

            pairs.append({
                "error_message": error_msg,
                "code_before": code_before,
                "code_after": code_after,
                "commit_message": commit_msg,
            })

        return pairs

    def get_final_status(self, workspace_dir: str) -> dict:
        """Parse FINAL: line from results.log."""
        results_log = Path(workspace_dir) / "results.log"
        if not results_log.exists():
            return {"passed": False, "attempts": 0}

        content = results_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"FINAL:\s*(PASS|FAIL)\s+attempts=(\d+)", content)
        if match:
            return {
                "passed": match.group(1) == "PASS",
                "attempts": int(match.group(2)),
            }
        # Fallback: check if any PASSED appears
        return {
            "passed": "PASSED" in content or "passed" in content,
            "attempts": 0,
        }

    def _parse_results_log(self, path: Path) -> list[str]:
        """Extract error messages from results.log.

        Returns a list of error messages, one per test run (between attempts).
        """
        if not path.exists():
            return []

        content = path.read_text(encoding="utf-8", errors="replace")
        errors = []
        current_error = []

        for line in content.split("\n"):
            if re.search(r"FAILED|Error|Traceback|AssertionError|RuntimeError|ImportError", line):
                current_error.append(line.strip())
            elif re.search(r"PASSED|passed", line):
                if current_error:
                    errors.append("\n".join(current_error))
                    current_error = []
                errors.append("")  # empty = this attempt passed

        if current_error:
            errors.append("\n".join(current_error))

        return errors

    @staticmethod
    def _git_show(ws: Path, sha: str, filename: str) -> str:
        """Get file content at a specific commit."""
        try:
            result = subprocess.run(
                ["git", "show", f"{sha}:{filename}"],
                cwd=ws, capture_output=True, text=True, check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    @staticmethod
    def _git_commit_msg(ws: Path, sha: str) -> str:
        """Get commit message."""
        try:
            result = subprocess.run(
                ["git", "log", "--format=%s", "-1", sha],
                cwd=ws, capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_git_trace_parser.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add eval/cutile/git_trace_parser.py tests/test_git_trace_parser.py
git commit -m "feat(eval): add GitTraceParser for extracting error→fix pairs from git history

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Autoresearch Solver

**Files:**
- Create: `eval/cutile/autoresearch_solver.py`

- [ ] **Step 1: Implement**

```python
# eval/cutile/autoresearch_solver.py
"""Autoresearch-style cuTile solver: program.md-driven OpenCode in Docker."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from .tilegym_loader import TileGymLoader
from .git_trace_parser import GitTraceParser

logger = logging.getLogger(__name__)

_DEFAULT_OPENCODE_IMAGE = "local/opencode-agent:cutile-synthv2"
_PROGRAM_MD = Path(__file__).parent / "program.md"


class AutoresearchSolver:
    """Solve TileGym kernels using autoresearch pattern with program.md."""

    def __init__(
        self,
        tilegym_dir: str,
        skill_dir: str,
        opencode_image: str = _DEFAULT_OPENCODE_IMAGE,
        timeout: int = 600,
        model: str = "nvidia/aws/anthropic/bedrock-claude-opus-4-6",
    ):
        self._loader = TileGymLoader(tilegym_dir)
        self._tilegym_dir = Path(tilegym_dir).resolve()
        self._skill_dir = Path(skill_dir).resolve()
        self._opencode_image = opencode_image
        self._timeout = timeout
        self._model = model
        self._trace_parser = GitTraceParser()

    def _create_workspace(self, kernel_name: str, workspace_dir: Path) -> None:
        """Create a solve workspace with program.md, problem spec, test, and skill."""
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True)

        # 1. Copy program.md
        shutil.copy2(_PROGRAM_MD, workspace_dir / "program.md")

        # 2. Write problem spec
        spec = {
            "kernel": kernel_name,
            "description": f"Implement the {kernel_name} cuTile kernel that passes the test in test/",
        }
        (workspace_dir / "problem-spec.yaml").write_text(
            json.dumps(spec, indent=2), encoding="utf-8"
        )

        # 3. Copy test file (VISIBLE to agent)
        test_dir = workspace_dir / "test"
        test_dir.mkdir()
        test_source = self._loader.get_test_source(kernel_name)
        if test_source:
            (test_dir / f"test_{kernel_name}.py").write_text(test_source, encoding="utf-8")

        # Copy test utilities
        for fname in ["common.py", "conftest.py", "__init__.py"]:
            src = self._tilegym_dir / "tests" / fname
            if src.exists():
                (test_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        # Also create tests/__init__.py for import resolution
        tests_pkg = test_dir.parent / "tests"
        if not tests_pkg.exists():
            tests_pkg.mkdir(exist_ok=True)
            (tests_pkg / "__init__.py").write_text("", encoding="utf-8")

        # 4. Copy skill dir into workspace
        skill_dest = workspace_dir / ".opencode" / "skill" / "cutile-python"
        shutil.copytree(self._skill_dir, skill_dest)

        # 5. Create empty results.log
        (workspace_dir / "results.log").write_text("", encoding="utf-8")

    def solve_kernel(self, kernel_name: str, output_dir: Path) -> dict:
        """Solve a single kernel. Returns result dict."""
        workspace = output_dir / "workspaces" / kernel_name
        self._create_workspace(kernel_name, workspace)

        # Build OpenCode config
        opencode_config = json.dumps({
            "model": self._model,
            "small_model": self._model,
            "provider": {
                "nvidia": {
                    "name": "nvidia",
                    "npm": "@ai-sdk/openai-compatible",
                    "models": {
                        "aws/anthropic/bedrock-claude-opus-4-6": {
                            "name": self._model,
                            "id": "aws/anthropic/bedrock-claude-opus-4-6",
                        }
                    },
                    "options": {
                        "apiKey": os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "")),
                        "baseURL": "https://inference-api.nvidia.com/v1",
                    },
                }
            },
        }, separators=(",", ":"))

        # Launch OpenCode in Docker
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    "--entrypoint", "",
                    "-v", f"{workspace}:/testbed:rw",
                    "-e", f"OPENCODE_CONFIG_CONTENT={opencode_config}",
                    "-e", f"OPENAI_API_KEY={os.environ.get('API_KEY', os.environ.get('OPENAI_API_KEY', ''))}",
                    "-e", "CUDA_TILE_CACHE_DIR=/tmp/cutile-cache",
                    "-e", "XDG_DATA_HOME=/tmp/xdg-data",
                    "-e", "XDG_CONFIG_HOME=/tmp/xdg-config",
                    "-e", "XDG_STATE_HOME=/tmp/xdg-state",
                    "-w", "/testbed",
                    self._opencode_image,
                    "opencode", "run",
                    "-m", self._model,
                    "--dir", "/testbed",
                    "Follow the instructions in program.md",
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired:
            logger.warning("[AutoSolver] %s: TIMEOUT after %ds", kernel_name, self._timeout)
            stdout = ""
            stderr = "TIMEOUT"
        except Exception as e:
            logger.error("[AutoSolver] %s: ERROR %s", kernel_name, e)
            stdout = ""
            stderr = str(e)

        # Parse results
        status = self._trace_parser.get_final_status(str(workspace))
        pairs = self._trace_parser.parse(str(workspace))

        logger.info(
            "[AutoSolver] %s: %s (attempts=%d, pairs=%d)",
            kernel_name,
            "SOLVED" if status["passed"] else "UNSOLVED",
            status["attempts"],
            len(pairs),
        )

        return {
            "kernel_name": kernel_name,
            "passed": status["passed"],
            "attempts": status["attempts"],
            "num_pairs": len(pairs),
            "pairs": pairs,
            "workspace": str(workspace),
            "stderr_tail": stderr[-500:] if stderr else "",
        }

    def solve_all(self, output_dir: str) -> list[dict]:
        """Solve all TileGym kernels. Returns list of results."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        kernels = self._loader.list_kernels()
        results = []

        for i, kernel_name in enumerate(kernels):
            logger.info("[AutoSolver] Solving %s (%d/%d)", kernel_name, i + 1, len(kernels))
            result = self.solve_kernel(kernel_name, output)
            results.append(result)

            # Save incrementally
            with open(output / "results.jsonl", "a") as f:
                # Don't write pairs to results file (too large)
                r = {k: v for k, v in result.items() if k != "pairs"}
                f.write(json.dumps(r) + "\n")

        solved = sum(1 for r in results if r["passed"])
        logger.info("[AutoSolver] Done: %d/%d solved", solved, len(results))
        return results
```

- [ ] **Step 2: Verify import**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -c "from eval.cutile.autoresearch_solver import AutoresearchSolver; print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add eval/cutile/autoresearch_solver.py
git commit -m "feat(eval): add AutoresearchSolver — program.md-driven OpenCode in Docker

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Update run_skill_synthesis.py

**Files:**
- Modify: `eval/cutile/run_skill_synthesis.py`

- [ ] **Step 1: Rewrite to use AutoresearchSolver**

```python
# eval/cutile/run_skill_synthesis.py
"""Main entry point for autoresearch-style skill synthesis."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.cutile.autoresearch_solver import AutoresearchSolver
from eval.cutile.skill_synthesizer import SkillSynthesizer
from eval.cutile.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Autoresearch-style cuTile Skill Synthesis")
    parser.add_argument("--tilegym-dir", required=True, help="Path to TileGym repo")
    parser.add_argument("--results-dir", default="eval/cutile/synthesis-results", help="Output dir")
    parser.add_argument("--base-skill", default="", help="Path to base cutile-python skill dir")
    parser.add_argument("--model", default="aws/anthropic/bedrock-claude-opus-4-6")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per kernel (seconds)")
    parser.add_argument("--skip-solve", action="store_true", help="Skip solving, use existing results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Solve ---
    all_pairs = []

    if not args.skip_solve:
        logger.info("=== Step 1: Autoresearch Solving ===")
        skill_dir = args.base_skill or str(results_dir / "base-skill")
        solver = AutoresearchSolver(
            tilegym_dir=args.tilegym_dir,
            skill_dir=skill_dir,
            timeout=args.timeout,
            model=f"nvidia/{args.model}",
        )
        solve_results = solver.solve_all(str(results_dir / "solve"))
        solved = sum(1 for r in solve_results if r["passed"])
        logger.info("Solved: %d/%d kernels", solved, len(solve_results))

        # Collect pairs from successful solves only
        for result in solve_results:
            if result["passed"] and result["pairs"]:
                for pair in result["pairs"]:
                    pair["kernel_name"] = result["kernel_name"]
                    all_pairs.append(pair)
    else:
        logger.info("=== Step 1: Skipped (loading existing results) ===")
        pairs_file = results_dir / "error_fix_pairs.jsonl"
        if pairs_file.exists():
            with open(pairs_file) as f:
                all_pairs = [json.loads(l) for l in f if l.strip()]

    logger.info("Total error→fix pairs: %d", len(all_pairs))

    # Save pairs
    with open(results_dir / "error_fix_pairs.jsonl", "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # --- Step 2: Synthesize Skills ---
    logger.info("=== Step 2: Synthesizing Skills ===")
    llm = LLMClient(model=args.model)
    skill_dir = str(results_dir / "evolved-skills")
    synthesizer = SkillSynthesizer(llm=llm, skill_dir=skill_dir)
    synth_result = synthesizer.run(all_pairs, output_dir=str(results_dir))
    logger.info("Synthesized: %d notes → %d skills", synth_result["num_notes"], synth_result["num_skills"])

    # --- Step 3: Merge Skills ---
    if args.base_skill:
        logger.info("=== Step 3: Merging Skills ===")
        merged_dir = results_dir / "cutile-python-synthesized"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        shutil.copytree(args.base_skill, str(merged_dir))

        # Append evolved skills to SKILL.md
        evolved_skills = list(synthesizer._skill_manager._iter_all_skills())
        skill_md = merged_dir / "SKILL.md"
        with open(skill_md, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n# Evolved Skills (from debug journeys)\n\n")
            for skill in evolved_skills:
                name = skill.get("name", "unnamed")
                desc = skill.get("description", "")
                content = skill.get("content", "")
                f.write(f"## {name}\n_{desc}_\n\n{content}\n\n")

        logger.info("Merged skill dir: %s", merged_dir)

    # --- Summary ---
    logger.info("=== Summary ===")
    if not args.skip_solve:
        logger.info("  Kernels solved: %d/%d", solved, len(solve_results))
    logger.info("  Error→fix pairs: %d", len(all_pairs))
    logger.info("  Teaching notes: %d", synth_result["num_notes"])
    logger.info("  Skills evolved: %d", synth_result["num_skills"])
    logger.info("  Results dir: %s", results_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python eval/cutile/run_skill_synthesis.py --help`
Expected: Prints usage

- [ ] **Step 3: Commit**

```bash
git add eval/cutile/run_skill_synthesis.py
git commit -m "refactor(eval): update run_skill_synthesis to use AutoresearchSolver

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Update B200 Run Scripts

**Files:**
- Modify: `runs/run-synthesis-b200.sh`

- [ ] **Step 1: Update run script**

```bash
#!/bin/bash
# runs/run-synthesis-b200.sh
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
TILEGYM="$WORKDIR/TileGym"
BASE_SKILL="$WORKDIR/skills/cutile-python"
RESULTS="$WORKDIR/eval/cutile/synthesis-results"

export API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export OPENAI_API_KEY="$API_KEY"

# Load Docker images
OPENCODE_IMAGE="local/opencode-agent:cutile-synthv2"
OPENCODE_TAR="$WORKDIR/runs/opencode-agent-cutile-synthv2.tar"
if ! docker image inspect "$OPENCODE_IMAGE" &>/dev/null; then
  if [[ -f "$OPENCODE_TAR" ]]; then
    echo "Loading OpenCode agent image..."
    docker load -i "$OPENCODE_TAR"
  else
    echo "ERROR: OpenCode image not found. Build it first."
    exit 1
  fi
fi

echo "=== Autoresearch-Style Skill Synthesis ==="
echo "TileGym: $TILEGYM"
echo "Base skill: $BASE_SKILL"
echo "Results: $RESULTS"

# Clean previous results
rm -rf "$RESULTS"

# Install openai if needed
pip install openai 2>/dev/null || true

cd "$WORKDIR"
python eval/cutile/run_skill_synthesis.py \
  --tilegym-dir "$TILEGYM" \
  --results-dir "$RESULTS" \
  --base-skill "$BASE_SKILL" \
  --model aws/anthropic/bedrock-claude-opus-4-6 \
  --timeout 600

echo "=== Done ==="
echo "Skills: $RESULTS/cutile-python-synthesized/"
ls "$RESULTS/cutile-python-synthesized/SKILL.md" 2>/dev/null && echo "Merged SKILL.md exists" || echo "WARNING: No merged skill"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x runs/run-synthesis-b200.sh
git add runs/run-synthesis-b200.sh
git commit -m "feat(eval): update B200 run script for autoresearch solver

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Full Test Suite + Push

- [ ] **Step 1: Run all tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/ -v --ignore=tests/test_v03_live_tinker.py --ignore=tests/test_setup_wizard.py`
Expected: All tests pass

- [ ] **Step 2: Push**

```bash
git push origin kaix/conversation-driven-skill-evolution
```
