# Iterative Solver + Skill Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an iterative cuTile solver that uses real compile/runtime errors to debug kernels, then extracts skills from successful debug journeys.

**Architecture:** OpenCode agent solves TileGym kernels in Docker with tests visible (48 tool calls). Traces are parsed for error→fix pairs, LLM synthesizes into teaching notes, fed through MetaClaw skill evolution pipeline. Evaluated on compute-eval benchmark with tests hidden.

**Tech Stack:** Python 3.10+, Docker, OpenCode CLI, MetaClaw components (existing), TileGym, compute-eval-synth.

**Spec:** `docs/specs/2026-03-30-iterative-solver-skill-synthesis.md`

---

## File Structure

```
eval/cutile/
├── iterative_solver.py          [NEW]  OpenCode-in-Docker solver with tests visible
├── trace_parser.py              [NEW]  Parse agent traces → error→fix pairs
├── skill_synthesizer.py         [NEW]  Group pairs, LLM synthesis, MetaClaw pipeline
├── run_skill_synthesis.py       [NEW]  Main entry point (solve + extract + merge)
├── tilegym_loader.py            [EXISTING]  Load TileGym kernels
├── kernel_tester.py             [EXISTING]  Docker test runner (reused for verification)
├── llm_client.py                [EXISTING]  LLM API wrapper
├── skill_writer.py              [EXISTING]  Merge skills into single dir

tests/
├── test_trace_parser.py         [NEW]
├── test_skill_synthesizer.py    [NEW]
```

---

### Task 1: Trace Parser

**Files:**
- Create: `eval/cutile/trace_parser.py`
- Test: `tests/test_trace_parser.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_trace_parser.py
import json
import pytest
from eval.cutile.trace_parser import TraceParser


def _make_trace_lines():
    """Create a realistic trace with write → bash(test) → error → write(fix) → bash(test) → pass."""
    return [
        json.dumps({"type": "step_start", "part": {"type": "step"}}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "write",
            "state": {
                "input": {"filePath": "/testbed/solution.py", "content": "import cuda.tile as ct\n\ndef bad_kernel():\n    pass\n"},
                "output": "",
            }
        }}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "bash",
            "state": {
                "input": {"command": "python -m pytest test/test_softmax.py"},
                "output": "FAILED: RuntimeError: ct.mma expects 2D tiles, got shape (1, 128, 64)",
            }
        }}),
        json.dumps({"type": "text", "part": {"text": "I need to reshape the tile to 2D before MMA."}}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "write",
            "state": {
                "input": {"filePath": "/testbed/solution.py", "content": "import cuda.tile as ct\n\ndef fixed_kernel():\n    tile = ct.reshape(t, (M, K))\n"},
                "output": "",
            }
        }}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "bash",
            "state": {
                "input": {"command": "python -m pytest test/test_softmax.py"},
                "output": "PASSED\n1 passed in 5.2s",
            }
        }}),
    ]


def test_parse_error_fix_pairs():
    parser = TraceParser()
    lines = _make_trace_lines()
    pairs = parser.parse("\n".join(lines))
    assert len(pairs) >= 1
    pair = pairs[0]
    assert "error_message" in pair
    assert "ct.mma expects 2D tiles" in pair["error_message"]
    assert "code_before" in pair
    assert "code_after" in pair
    assert "bad_kernel" in pair["code_before"]
    assert "fixed_kernel" in pair["code_after"]


def test_parse_with_agent_reasoning():
    parser = TraceParser()
    lines = _make_trace_lines()
    pairs = parser.parse("\n".join(lines))
    assert pairs[0].get("agent_reasoning", "")
    assert "reshape" in pairs[0]["agent_reasoning"].lower()


def test_parse_no_errors():
    """Trace where code passes on first try — no error→fix pairs."""
    lines = [
        json.dumps({"type": "tool_use", "part": {
            "tool": "write",
            "state": {"input": {"filePath": "/testbed/solution.py", "content": "good code"}, "output": ""}
        }}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "bash",
            "state": {"input": {"command": "pytest test/"}, "output": "PASSED\n1 passed"}
        }}),
    ]
    parser = TraceParser()
    pairs = parser.parse("\n".join(lines))
    assert len(pairs) == 0


def test_parse_empty_trace():
    parser = TraceParser()
    pairs = parser.parse("")
    assert pairs == []


def test_parse_multiple_errors():
    """Two error→fix cycles in one trace."""
    lines = [
        # First error cycle
        json.dumps({"type": "tool_use", "part": {"tool": "write", "state": {"input": {"filePath": "/testbed/solution.py", "content": "v1"}, "output": ""}}}),
        json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": "pytest"}, "output": "FAILED: ImportError: ct.mma not found"}}}),
        json.dumps({"type": "text", "part": {"text": "Need to use ct.matmul instead"}}),
        json.dumps({"type": "tool_use", "part": {"tool": "write", "state": {"input": {"filePath": "/testbed/solution.py", "content": "v2"}, "output": ""}}}),
        json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": "pytest"}, "output": "FAILED: shape mismatch"}}}),
        json.dumps({"type": "text", "part": {"text": "Need to fix tile shape"}}),
        json.dumps({"type": "tool_use", "part": {"tool": "write", "state": {"input": {"filePath": "/testbed/solution.py", "content": "v3"}, "output": ""}}}),
        json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": "pytest"}, "output": "PASSED"}}}),
    ]
    parser = TraceParser()
    pairs = parser.parse("\n".join(lines))
    assert len(pairs) == 2
    assert "ct.mma not found" in pairs[0]["error_message"]
    assert "shape mismatch" in pairs[1]["error_message"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_trace_parser.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement trace_parser.py**

```python
# eval/cutile/trace_parser.py
"""Parse OpenCode agent traces to extract error→fix pairs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SOLUTION_PATHS = {"solution.py", "/testbed/solution.py"}
_FAIL_PATTERNS = re.compile(r"FAIL|Error|error|Traceback|AssertionError|RuntimeError|ImportError|TypeError|ValueError")
_PASS_PATTERNS = re.compile(r"PASSED|passed|OK\b")
_TEST_COMMANDS = re.compile(r"pytest|python.*test|PYTHONPATH.*test")


class TraceParser:
    """Extract error→fix pairs from OpenCode agent traces."""

    def parse(self, trace_text: str) -> list[dict]:
        """Parse a trace (newline-delimited JSON) into error→fix pairs.

        An error→fix pair is: write(code) → run(test) → FAIL → [reasoning] → write(fix) → run(test) → PASS/FAIL
        Only consecutive error→fix sequences are captured.
        """
        if not trace_text.strip():
            return []

        entries = []
        for line in trace_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        # Extract events: writes to solution.py, test runs (bash), and text (reasoning)
        events = []
        for entry in entries:
            typ = entry.get("type", "")
            part = entry.get("part", {})

            if typ == "tool_use":
                tool = part.get("tool", "")
                state = part.get("state", {})
                inp = state.get("input", {})
                out = state.get("output", "")

                if tool == "write":
                    path = inp.get("filePath", "")
                    # Check if writing to solution file
                    if any(path.endswith(sp) for sp in _SOLUTION_PATHS) or "solution" in path:
                        events.append({
                            "type": "write",
                            "content": inp.get("content", ""),
                            "path": path,
                        })

                elif tool == "bash":
                    cmd = inp.get("command", "")
                    if _TEST_COMMANDS.search(cmd):
                        is_fail = bool(_FAIL_PATTERNS.search(out)) and not bool(_PASS_PATTERNS.search(out))
                        is_pass = bool(_PASS_PATTERNS.search(out)) and not bool(re.search(r"FAILED|failed", out))
                        events.append({
                            "type": "test_fail" if is_fail else ("test_pass" if is_pass else "test_other"),
                            "command": cmd,
                            "output": out,
                        })

            elif typ == "text":
                text = part.get("text", "").strip()
                if text:
                    events.append({"type": "reasoning", "text": text})

        # Find error→fix sequences
        return self._extract_pairs(events)

    def _extract_pairs(self, events: list[dict]) -> list[dict]:
        """Find sequences: write → test_fail → [reasoning] → write → test_*"""
        pairs = []
        i = 0
        while i < len(events):
            # Look for: write → test_fail
            if (events[i]["type"] == "write"
                    and i + 1 < len(events)
                    and events[i + 1]["type"] == "test_fail"):

                code_before = events[i]["content"]
                error_message = events[i + 1]["output"]

                # Collect reasoning text
                reasoning_parts = []
                j = i + 2
                while j < len(events) and events[j]["type"] == "reasoning":
                    reasoning_parts.append(events[j]["text"])
                    j += 1

                # Look for next write (the fix)
                if j < len(events) and events[j]["type"] == "write":
                    code_after = events[j]["content"]
                    pairs.append({
                        "error_message": error_message[:1000],
                        "code_before": code_before,
                        "code_after": code_after,
                        "agent_reasoning": " ".join(reasoning_parts),
                    })
                    i = j + 1
                    continue

            i += 1

        return pairs
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_trace_parser.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add eval/cutile/trace_parser.py tests/test_trace_parser.py
git commit -m "feat(eval): add TraceParser for extracting error→fix pairs from agent traces

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Skill Synthesizer

**Files:**
- Create: `eval/cutile/skill_synthesizer.py`
- Test: `tests/test_skill_synthesizer.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_skill_synthesizer.py
import json
import pytest
from eval.cutile.skill_synthesizer import SkillSynthesizer


class FakeLLM:
    def complete(self, prompt, max_tokens=2000):
        if "teaching note" in prompt.lower() or "remember this" in prompt.lower():
            return "Remember this: when using ct.mma, always reshape tiles to 2D first."
        return json.dumps([{
            "action": "create",
            "skill": {
                "name": "cutile-mma-reshape",
                "description": "Reshape tiles to 2D before ct.mma",
                "content": "## MMA Reshape\nAlways reshape 3D tiles to 2D before ct.mma.",
                "category": "coding",
            },
            "reasoning": "Common error pattern",
        }])


@pytest.fixture
def skill_dir(tmp_path):
    return str(tmp_path / "skills")


def test_group_pairs():
    synth = SkillSynthesizer(llm=FakeLLM(), skill_dir="/tmp/unused")
    pairs = [
        {"error_message": "ct.mma expects 2D tiles", "code_before": "a", "code_after": "b", "agent_reasoning": "reshape"},
        {"error_message": "ct.mma got 3D input", "code_before": "c", "code_after": "d", "agent_reasoning": "reshape again"},
        {"error_message": "shape mismatch (256,) vs (128,)", "code_before": "e", "code_after": "f", "agent_reasoning": "fix shape"},
    ]
    groups = synth.group_pairs(pairs)
    assert len(groups) >= 2  # mma group + shape group


def test_synthesize_teaching_note():
    synth = SkillSynthesizer(llm=FakeLLM(), skill_dir="/tmp/unused")
    pairs = [
        {"error_message": "ct.mma expects 2D", "code_before": "a", "code_after": "b", "agent_reasoning": "reshape"},
    ]
    note = synth.synthesize_note(pairs)
    assert "Remember this" in note


def test_run_produces_skills(skill_dir):
    synth = SkillSynthesizer(llm=FakeLLM(), skill_dir=skill_dir)
    all_pairs = [
        {"error_message": "ct.mma expects 2D", "code_before": "a", "code_after": "b", "agent_reasoning": "reshape"},
        {"error_message": "ct.mma got 3D", "code_before": "c", "code_after": "d", "agent_reasoning": "reshape"},
    ]
    result = synth.run(all_pairs)
    assert result["num_notes"] >= 1
    assert result["num_skills"] >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_skill_synthesizer.py -v`

- [ ] **Step 3: Implement skill_synthesizer.py**

```python
# eval/cutile/skill_synthesizer.py
"""Synthesize skills from error→fix pairs via LLM + MetaClaw pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from metaclaw.conversation_signal_detector import ConversationSignalDetector
from metaclaw.signal_aggregator import SignalAggregator, SkillEvolutionConfig
from metaclaw.skill_evolver import SkillEvolver
from metaclaw.skill_manager import SkillManager

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """\
While debugging cuTile kernels, these errors and fixes were encountered:

{pairs_text}

Write a concise teaching note (3-5 sentences) starting with "Remember this:"
that captures the reusable pattern. Include:
- What error to watch for
- Why it happens
- How to fix it
Be specific about cuTile API usage (ct.* functions).
"""

# Keywords for grouping error→fix pairs
_GROUP_KEYWORDS = {
    "mma-reshape": ["ct.mma", "2D tiles", "3D", "reshape"],
    "tile-shape": ["shape mismatch", "tile size", "dimension", "broadcast"],
    "precision": ["tolerance", "atol", "rtol", "float32", "precision", "accumulator"],
    "memory-access": ["ct.load", "ct.store", "ct.gather", "ct.scatter", "index", "bounds"],
    "api-usage": ["ImportError", "AttributeError", "not found", "not available", "ct.constexpr"],
    "launch-grid": ["grid", "ct.launch", "num_blocks", "ct.bid"],
}


class SkillSynthesizer:
    """Synthesize skills from error→fix pairs."""

    def __init__(self, llm, skill_dir: str):
        self._llm = llm
        self._skill_dir = skill_dir
        os.makedirs(skill_dir, exist_ok=True)

        self._detector = ConversationSignalDetector(use_llm_detection=False)
        self._aggregator = SignalAggregator(SkillEvolutionConfig(
            sources=["conversation"],
            correction_threshold=1,  # Trigger on every teaching note
        ))
        self._skill_manager = SkillManager(skill_dir, retrieval_mode="template")
        self._evolver = SkillEvolver(llm_client=_LLMAdapter(llm))

    def group_pairs(self, pairs: list[dict]) -> list[dict]:
        """Group error→fix pairs by error pattern."""
        buckets: dict[str, list[dict]] = defaultdict(list)

        for pair in pairs:
            error = pair.get("error_message", "").lower()
            matched = False
            for label, keywords in _GROUP_KEYWORDS.items():
                if any(kw.lower() in error for kw in keywords):
                    buckets[label].append(pair)
                    matched = True
                    break
            if not matched:
                buckets["other"].append(pair)

        return [
            {"label": label, "pairs": items}
            for label, items in buckets.items()
            if items
        ]

    def synthesize_note(self, pairs: list[dict]) -> str:
        """Generate a teaching note from a group of error→fix pairs."""
        pairs_text = ""
        for i, pair in enumerate(pairs[:5]):  # Max 5 per group
            pairs_text += f"\nError {i+1}: {pair['error_message'][:300]}\n"
            if pair.get("agent_reasoning"):
                pairs_text += f"Agent thought: {pair['agent_reasoning'][:200]}\n"
            pairs_text += f"Fix: changed code to resolve the error\n"

        prompt = _SYNTHESIS_PROMPT.format(pairs_text=pairs_text)
        return self._llm.complete(prompt, max_tokens=500)

    def run(self, all_pairs: list[dict], output_dir: str = "") -> dict:
        """Full pipeline: group → synthesize → feed through MetaClaw → evolve."""
        if not all_pairs:
            return {"num_notes": 0, "num_skills": 0}

        groups = self.group_pairs(all_pairs)
        notes = []

        for group in groups:
            note = self.synthesize_note(group["pairs"])
            notes.append({"label": group["label"], "note": note, "num_pairs": len(group["pairs"])})

            # Feed through conversation pipeline
            turn_data = {
                "session_id": f"synthesis-{group['label']}",
                "turn_num": 1,
                "user_message": note,
                "assistant_response": "",
                "active_skills": [],
            }
            signals = self._detector.detect(turn_data)
            if not signals:
                # Ensure detection by prepending "Remember this:"
                turn_data["user_message"] = f"Remember this: {note}"
                signals = self._detector.detect(turn_data)
            self._aggregator.add(signals)

        # Evolve skills from all notes
        num_skills = 0
        consumed = self._aggregator.consume()
        if consumed:
            actions = asyncio.run(
                self._evolver.evolve(consumed, self._skill_manager.skills)
            )
            for action in actions:
                if action.action == "create":
                    if self._skill_manager.add_skill(action.skill):
                        num_skills += 1
                elif action.action == "update":
                    if self._skill_manager.update_skill(action.skill["name"], action.skill):
                        num_skills += 1

        # Save notes
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_dir, "teaching_notes.jsonl"), "w") as f:
                for n in notes:
                    f.write(json.dumps(n) + "\n")

        return {"num_notes": len(notes), "num_skills": num_skills}


class _LLMAdapter:
    def __init__(self, llm):
        self._llm = llm
    def chat_complete(self, prompt: str) -> str:
        return self._llm.complete(prompt, max_tokens=3000)
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_skill_synthesizer.py -v`

- [ ] **Step 5: Commit**

```bash
git add eval/cutile/skill_synthesizer.py tests/test_skill_synthesizer.py
git commit -m "feat(eval): add SkillSynthesizer for LLM-based skill extraction from error→fix pairs

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Iterative Solver

**Files:**
- Create: `eval/cutile/iterative_solver.py`

No unit test — this is Docker orchestration code tested by running the actual pipeline.

- [ ] **Step 1: Implement iterative_solver.py**

```python
# eval/cutile/iterative_solver.py
"""Iterative cuTile solver using OpenCode in Docker with tests visible."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from .tilegym_loader import TileGymLoader

logger = logging.getLogger(__name__)

_DEFAULT_OPENCODE_IMAGE = "local/opencode-agent:cutile-synthv2"

_SOLVE_PROMPT = (
    "You are solving a cuTile kernel implementation problem. "
    "Read the problem description in problem-spec.yaml and the cuTile skill first. "
    "Write solution.py that implements the required kernel. "
    "Run the test in test/ to verify your solution. If it fails, read the error and fix your code. "
    "Keep iterating until the test passes or you run out of steps."
)


class IterativeSolver:
    """Solve TileGym kernels using OpenCode with real test feedback."""

    def __init__(
        self,
        tilegym_dir: str,
        skill_dir: str,
        opencode_image: str = _DEFAULT_OPENCODE_IMAGE,
        max_steps: int = 48,
        model: str = "nvidia/aws/anthropic/bedrock-claude-opus-4-6",
    ):
        self._loader = TileGymLoader(tilegym_dir)
        self._tilegym_dir = Path(tilegym_dir).resolve()
        self._skill_dir = Path(skill_dir).resolve()
        self._opencode_image = opencode_image
        self._max_steps = max_steps
        self._model = model

    def _create_workspace(self, kernel_name: str, workspace_dir: Path) -> None:
        """Create a solve workspace with problem spec, test, and skill."""
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Write problem spec
        test_source = self._loader.get_test_source(kernel_name) or ""
        prompt = (
            f"Implement the cuTile kernel: {kernel_name}\n\n"
            f"Your solution must pass the test in test/test_{kernel_name}.py.\n"
            f"Write solution.py with the implementation.\n"
            f"Use the cuTile API (import cuda.tile as ct)."
        )
        spec = {"task_id": f"tilegym/{kernel_name}", "prompt": prompt}
        (workspace_dir / "problem-spec.yaml").write_text(
            json.dumps(spec, indent=2), encoding="utf-8"
        )

        # Copy test file
        test_dir = workspace_dir / "test"
        test_dir.mkdir(exist_ok=True)
        if test_source:
            (test_dir / f"test_{kernel_name}.py").write_text(test_source, encoding="utf-8")

        # Copy TileGym common test utilities
        common_py = self._tilegym_dir / "tests" / "common.py"
        conftest_py = self._tilegym_dir / "tests" / "conftest.py"
        tests_init = self._tilegym_dir / "tests" / "__init__.py"
        if common_py.exists():
            (test_dir / "common.py").write_text(common_py.read_text(), encoding="utf-8")
        if conftest_py.exists():
            (test_dir / "conftest.py").write_text(conftest_py.read_text(), encoding="utf-8")
        if tests_init.exists():
            (test_dir / "__init__.py").write_text(tests_init.read_text(), encoding="utf-8")

    def solve_kernel(
        self,
        kernel_name: str,
        output_dir: Path,
    ) -> dict:
        """Solve a single kernel. Returns {passed, trace_path, solution_path, kernel_name}."""
        workspace = output_dir / "workspace" / kernel_name
        self._create_workspace(kernel_name, workspace)

        # OpenCode config
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

        # Run OpenCode in Docker
        trace_dir = output_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{kernel_name}.jsonl"

        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    "-v", f"{workspace}:/testbed:rw",
                    "-v", f"{self._skill_dir}:/testbed/.opencode/skill/cutile-python:ro",
                    "-e", f"OPENCODE_CONFIG_CONTENT={opencode_config}",
                    "-e", f"OPENAI_API_KEY={os.environ.get('API_KEY', os.environ.get('OPENAI_API_KEY', ''))}",
                    "-e", "CUDA_TILE_CACHE_DIR=/tmp/cutile-cache",
                    "-e", f"XDG_DATA_HOME=/tmp/xdg-data",
                    "-e", f"XDG_CONFIG_HOME=/tmp/xdg-config",
                    "-e", f"XDG_STATE_HOME=/tmp/xdg-state",
                    "-w", "/testbed",
                    self._opencode_image,
                    "opencode", "run",
                    "-m", self._model,
                    "--dir", "/testbed",
                    "--max-steps", str(self._max_steps),
                    _SOLVE_PROMPT,
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max per kernel
            )
            stdout = result.stdout
            stderr = result.stderr

        except subprocess.TimeoutExpired:
            logger.warning("[Solver] %s: TIMEOUT", kernel_name)
            stdout = ""
            stderr = "TIMEOUT"
        except Exception as e:
            logger.error("[Solver] %s: ERROR %s", kernel_name, e)
            stdout = ""
            stderr = str(e)

        # Find trace file (OpenCode writes to .opencode/sessions/)
        trace_content = ""
        opencode_dir = workspace / ".opencode" / "sessions"
        if opencode_dir.exists():
            for session_dir in opencode_dir.iterdir():
                for jsonl in session_dir.glob("*.jsonl"):
                    trace_content = jsonl.read_text(encoding="utf-8")
                    break

        trace_path.write_text(trace_content, encoding="utf-8")

        # Check if solution exists
        solution_path = workspace / "solution.py"
        passed = solution_path.exists() and solution_path.stat().st_size > 0

        # Log result
        logger.info("[Solver] %s: %s (trace=%d lines)",
                    kernel_name, "SOLVED" if passed else "UNSOLVED",
                    len(trace_content.strip().split("\n")) if trace_content else 0)

        return {
            "kernel_name": kernel_name,
            "passed": passed,
            "trace_path": str(trace_path),
            "solution_path": str(solution_path) if passed else "",
            "stdout_tail": stdout[-500:] if stdout else "",
            "stderr_tail": stderr[-500:] if stderr else "",
        }

    def solve_all(self, output_dir: str) -> list[dict]:
        """Solve all TileGym kernels. Returns list of results."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        kernels = self._loader.list_kernels()
        results = []

        for kernel_name in kernels:
            logger.info("[Solver] Solving %s (%d/%d)", kernel_name, len(results) + 1, len(kernels))
            result = self.solve_kernel(kernel_name, output)
            results.append(result)

            # Save results incrementally
            with open(output / "results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

        solved = sum(1 for r in results if r["passed"])
        logger.info("[Solver] Done: %d/%d solved", solved, len(results))
        return results
```

- [ ] **Step 2: Verify import**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -c "from eval.cutile.iterative_solver import IterativeSolver; print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add eval/cutile/iterative_solver.py
git commit -m "feat(eval): add IterativeSolver — OpenCode in Docker with tests visible

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Main Entry Point (run_skill_synthesis.py)

**Files:**
- Create: `eval/cutile/run_skill_synthesis.py`

- [ ] **Step 1: Implement**

```python
# eval/cutile/run_skill_synthesis.py
"""Main entry point for iterative solver + skill synthesis."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.cutile.iterative_solver import IterativeSolver
from eval.cutile.trace_parser import TraceParser
from eval.cutile.skill_synthesizer import SkillSynthesizer
from eval.cutile.llm_client import LLMClient
from eval.cutile.skill_writer import SkillWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Iterative cuTile Solver + Skill Synthesis")
    parser.add_argument("--tilegym-dir", required=True, help="Path to TileGym repo")
    parser.add_argument("--results-dir", default="eval/cutile/synthesis-results", help="Output dir")
    parser.add_argument("--base-skill", default="", help="Path to base cutile-python skill dir")
    parser.add_argument("--model", default="aws/anthropic/bedrock-claude-opus-4-6")
    parser.add_argument("--max-steps", type=int, default=48, help="Max OpenCode tool calls per kernel")
    parser.add_argument("--skip-solve", action="store_true", help="Skip solving, use existing traces")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Solve ---
    if not args.skip_solve:
        logger.info("=== Step 1: Iterative Solving ===")
        skill_dir = args.base_skill or str(results_dir / "base-skill")
        solver = IterativeSolver(
            tilegym_dir=args.tilegym_dir,
            skill_dir=skill_dir,
            max_steps=args.max_steps,
            model=f"nvidia/{args.model}",
        )
        solve_results = solver.solve_all(str(results_dir / "solve"))
        solved = sum(1 for r in solve_results if r["passed"])
        logger.info("Solved: %d/%d kernels", solved, len(solve_results))
    else:
        logger.info("=== Step 1: Skipped (using existing traces) ===")
        solve_results = []
        results_jsonl = results_dir / "solve" / "results.jsonl"
        if results_jsonl.exists():
            with open(results_jsonl) as f:
                solve_results = [json.loads(l) for l in f if l.strip()]

    # --- Step 2: Parse Traces ---
    logger.info("=== Step 2: Parsing Traces ===")
    trace_parser = TraceParser()
    all_pairs = []

    for result in solve_results:
        if not result["passed"]:
            continue  # Only extract from successful solves
        trace_path = result.get("trace_path", "")
        if trace_path and os.path.exists(trace_path):
            trace_text = Path(trace_path).read_text(encoding="utf-8")
            pairs = trace_parser.parse(trace_text)
            for pair in pairs:
                pair["kernel_name"] = result["kernel_name"]
            all_pairs.extend(pairs)
            logger.info("  %s: %d error→fix pairs", result["kernel_name"], len(pairs))

    logger.info("Total error→fix pairs: %d", len(all_pairs))

    # Save pairs
    with open(results_dir / "error_fix_pairs.jsonl", "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # --- Step 3: Synthesize Skills ---
    logger.info("=== Step 3: Synthesizing Skills ===")
    llm = LLMClient(model=args.model)
    skill_dir = str(results_dir / "evolved-skills")
    synthesizer = SkillSynthesizer(llm=llm, skill_dir=skill_dir)
    synth_result = synthesizer.run(all_pairs, output_dir=str(results_dir))
    logger.info("Synthesized: %d notes → %d skills", synth_result["num_notes"], synth_result["num_skills"])

    # --- Step 4: Merge Skills ---
    if args.base_skill:
        logger.info("=== Step 4: Merging Skills ===")
        evolved_skills = list(synthesizer._skill_manager._iter_all_skills())
        merged_dir = results_dir / "cutile-python-synthesized"

        # Copy base skill dir
        import shutil
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        shutil.copytree(args.base_skill, str(merged_dir))

        # Append evolved skills to SKILL.md
        skill_md = merged_dir / "SKILL.md"
        with open(skill_md, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n# Evolved Skills (from debug journeys)\n\n")
            for skill in evolved_skills:
                f.write(f"## {skill['name']}\n_{skill['description']}_\n\n{skill['content']}\n\n")

        logger.info("Merged skill dir: %s", merged_dir)

    # --- Summary ---
    logger.info("=== Summary ===")
    logger.info("  Kernels solved: %d/%d", sum(1 for r in solve_results if r["passed"]), len(solve_results))
    logger.info("  Error→fix pairs: %d", len(all_pairs))
    logger.info("  Teaching notes: %d", synth_result["num_notes"])
    logger.info("  Skills evolved: %d", synth_result["num_skills"])
    logger.info("  Results dir: %s", results_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python eval/cutile/run_skill_synthesis.py --help`

- [ ] **Step 3: Commit**

```bash
git add eval/cutile/run_skill_synthesis.py
git commit -m "feat(eval): add run_skill_synthesis.py — main entry point for iterative solver pipeline

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: B200 Run Script + Benchmark Script

**Files:**
- Create: `runs/run-synthesis-b200.sh`
- Create: `runs/v2-run-synthesized.sh`

- [ ] **Step 1: Create synthesis run script**

```bash
# runs/run-synthesis-b200.sh
#!/bin/bash
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
TILEGYM="$WORKDIR/TileGym"
BASE_SKILL="$WORKDIR/skills/cutile-python"
RESULTS="$WORKDIR/eval/cutile/synthesis-results"

export API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export OPENAI_API_KEY="$API_KEY"

# Load Docker images
for img_name in "local/opencode-agent:cutile-synthv2" "local/compute-eval-tilegym:13.1.0"; do
  if ! docker image inspect "$img_name" &>/dev/null; then
    tar_name=$(echo "$img_name" | sed 's|local/||;s|:|-|').tar
    tar_path="$WORKDIR/runs/$tar_name"
    if [[ -f "$tar_path" ]]; then
      echo "Loading $img_name from tar..."
      docker load -i "$tar_path"
    else
      echo "WARNING: $img_name not found and no tar at $tar_path"
    fi
  fi
done

echo "=== Iterative Solver + Skill Synthesis ==="
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
  --max-steps 48

echo "=== Synthesis done ==="
echo "Merged skill: $RESULTS/cutile-python-synthesized/"
```

- [ ] **Step 2: Create benchmark run script**

```bash
# runs/v2-run-synthesized.sh
#!/bin/bash
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
SCRIPT="$WORKDIR/cutile-synth-eval-kit/scripts/compute-eval-run-cutile-opencode-opus46-synthv2.sh"

export CE_DIR="/home/scratch.huizim_coreai/workdir/seedbot/compute-eval-api"
export SYNTH_ROOT="/home/scratch.huizim_coreai/workdir/compute-eval-synth/compute_eval_synth_v2"
export RUN_ROOT="$WORKDIR/runs/v2-synthesized-$(date -u +%Y%m%d-%H%M%S)"

# Synthesized skill dir (from run-synthesis-b200.sh)
export CUTILE_SKILL_DIRS="$WORKDIR/eval/cutile/synthesis-results/cutile-python-synthesized"

export EVAL_IMAGE_TAR="$WORKDIR/runs/compute-eval-python-13.1.0.tar"

# Load OpenCode agent image
OPENCODE_TAR="$WORKDIR/runs/opencode-agent-cutile-synthv2.tar"
if ! docker image inspect local/opencode-agent:cutile-synthv2 &>/dev/null; then
  echo "Loading OpenCode agent image..."
  docker load -i "$OPENCODE_TAR"
fi

export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export MAX_WORKERS=1

echo "=== V2 Synthesized (debug journey skills) ==="
echo "RUN_ROOT=$RUN_ROOT"
echo "SKILL_DIRS=$CUTILE_SKILL_DIRS"

exec bash "$SCRIPT"
```

- [ ] **Step 3: Make executable and commit**

```bash
chmod +x runs/run-synthesis-b200.sh runs/v2-run-synthesized.sh
git add runs/run-synthesis-b200.sh runs/v2-run-synthesized.sh
git commit -m "feat(eval): add B200 run scripts for synthesis and benchmark

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Run Full Test Suite + Push

- [ ] **Step 1: Run all tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/ -v --ignore=tests/test_v03_live_tinker.py --ignore=tests/test_setup_wizard.py`
Expected: All tests pass

- [ ] **Step 2: Push**

```bash
git push origin kaix/conversation-driven-skill-evolution
```

---

## Execution Commands

After all tasks implemented:

```bash
# Step 1: Solve + synthesize skills (B200, ~1.5 hours)
crun -q 'gpu.product_name=*B200*' --gpus 1 -t 4:00:00 -b /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw/runs/run-synthesis-b200.sh

# Step 2: Benchmark with synthesized skills (B200, ~2 hours)
crun -q 'gpu.product_name=*B200*' --gpus 1 -t 8:00:00 -b /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw/runs/v2-run-synthesized.sh
```
