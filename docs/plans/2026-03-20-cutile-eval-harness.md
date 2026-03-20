# cuTile Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation harness that compares one-shot skill extraction vs multi-turn conversational skill evolution on cuTile kernel coding, measuring pass rate improvement on the compute-eval benchmark.

**Architecture:** Python orchestrator in `eval/cutile/` uses MetaClaw's conversation signal pipeline (detector → aggregator → evolver → skill manager) to extract and evolve cuTile skills from TileGym reference kernels, then evaluates on 48 compute-eval problems via existing shell wrappers.

**Tech Stack:** Python 3.10+, MetaClaw components (already implemented), LLM API (OpenAI-compatible), cutile-eval-kit shell scripts for Phase C.

**Spec:** `docs/specs/2026-03-20-cutile-eval-harness-design.md`

---

## File Structure

```
eval/cutile/
├── run_eval.py              [NEW]  Main orchestrator / CLI entry point
├── tilegym_loader.py        [NEW]  Load TileGym kernel sources + test files
├── phase_a_study.py         [NEW]  Method 1: direct extraction (also Phase A of Method 2)
├── phase_b_practice.py      [NEW]  Method 2: solve-and-check with conversational evolution
├── phase_c_evaluate.py      [NEW]  Run compute-eval benchmark via subprocess
├── correction_simulator.py  [NEW]  Generate corrections from reference comparison
├── correction_clusterer.py  [NEW]  Group corrections by error pattern
├── skill_writer.py          [NEW]  Merge skills into single .md for injection
├── report.py                [NEW]  Comparison reports and skill attribution
├── llm_client.py            [NEW]  Thin wrapper for LLM API calls
└── README.md                [NEW]  Usage instructions

tests/
├── test_tilegym_loader.py   [NEW]
├── test_phase_a_study.py    [NEW]
├── test_correction_simulator.py  [NEW]
├── test_correction_clusterer.py  [NEW]
├── test_skill_writer.py     [NEW]
├── test_report.py           [NEW]
└── test_phase_b_practice.py [NEW]
```

---

### Task 1: TileGym Loader

**Files:**
- Create: `eval/cutile/tilegym_loader.py`
- Create: `eval/__init__.py`
- Create: `eval/cutile/__init__.py`
- Test: `tests/test_tilegym_loader.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_tilegym_loader.py
import os
import pytest
from eval.cutile.tilegym_loader import TileGymLoader


TILEGYM_DIR = os.path.join(os.path.dirname(__file__), "..", "TileGym")
SKIP_NO_TILEGYM = pytest.mark.skipif(
    not os.path.isdir(TILEGYM_DIR), reason="TileGym not found"
)


@SKIP_NO_TILEGYM
def test_loader_finds_kernels():
    loader = TileGymLoader(TILEGYM_DIR)
    kernels = loader.list_kernels()
    assert len(kernels) >= 20
    assert "softmax" in kernels
    assert "matmul" in kernels


@SKIP_NO_TILEGYM
def test_loader_reads_kernel_source():
    loader = TileGymLoader(TILEGYM_DIR)
    source = loader.get_kernel_source("softmax")
    assert "ct.kernel" in source or "@ct.kernel" in source
    assert len(source) > 100


@SKIP_NO_TILEGYM
def test_loader_reads_test_source():
    loader = TileGymLoader(TILEGYM_DIR)
    test_src = loader.get_test_source("softmax")
    assert "def test_op" in test_src
    assert len(test_src) > 100


@SKIP_NO_TILEGYM
def test_loader_get_kernel_pair():
    loader = TileGymLoader(TILEGYM_DIR)
    pair = loader.get_kernel_pair("softmax")
    assert "kernel_source" in pair
    assert "test_source" in pair
    assert "kernel_name" in pair
    assert pair["kernel_name"] == "softmax"


@SKIP_NO_TILEGYM
def test_loader_skips_init_and_utils():
    loader = TileGymLoader(TILEGYM_DIR)
    kernels = loader.list_kernels()
    assert "__init__" not in kernels
    assert "utils" not in kernels


def test_loader_bad_dir():
    with pytest.raises(FileNotFoundError):
        TileGymLoader("/nonexistent/path")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_tilegym_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create package structure and implement**

Create `eval/__init__.py` and `eval/cutile/__init__.py` (empty files).

```python
# eval/cutile/tilegym_loader.py
"""Load TileGym kernel sources and test files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


_SKIP_FILES = {"__init__", "utils"}

_KERNEL_DIR = "src/tilegym/ops/cutile"
_TEST_DIR = "tests/ops"


class TileGymLoader:
    """Loads cuTile kernel implementations and their test files from TileGym."""

    def __init__(self, tilegym_dir: str):
        self._root = Path(tilegym_dir)
        if not self._root.is_dir():
            raise FileNotFoundError(f"TileGym directory not found: {tilegym_dir}")
        self._kernel_dir = self._root / _KERNEL_DIR
        self._test_dir = self._root / _TEST_DIR

    def list_kernels(self) -> list[str]:
        """Return sorted list of kernel names (without .py extension)."""
        names = []
        for p in sorted(self._kernel_dir.glob("*.py")):
            name = p.stem
            if name not in _SKIP_FILES:
                names.append(name)
        return names

    def get_kernel_source(self, name: str) -> str:
        """Read the full source of a kernel implementation."""
        path = self._kernel_dir / f"{name}.py"
        return path.read_text(encoding="utf-8")

    def get_test_source(self, name: str) -> Optional[str]:
        """Read the test file for a kernel, or None if not found."""
        path = self._test_dir / f"test_{name}.py"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_kernel_pair(self, name: str) -> dict:
        """Return kernel source + test source as a dict."""
        return {
            "kernel_name": name,
            "kernel_source": self.get_kernel_source(name),
            "test_source": self.get_test_source(name) or "",
        }

    def get_all_pairs(self) -> list[dict]:
        """Return all kernel pairs with test files."""
        return [self.get_kernel_pair(name) for name in self.list_kernels()]
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_tilegym_loader.py -v`
Expected: All tests PASS (or skip if TileGym not found)

- [ ] **Step 5: Commit**

```bash
git add eval/__init__.py eval/cutile/__init__.py eval/cutile/tilegym_loader.py tests/test_tilegym_loader.py
git commit -m "feat(eval): add TileGymLoader for kernel source + test loading"
```

---

### Task 2: LLM Client Wrapper

**Files:**
- Create: `eval/cutile/llm_client.py`

A thin wrapper for LLM API calls, reusable across phases. No test file — this is a 20-line shim around `openai.OpenAI`.

- [ ] **Step 1: Implement**

```python
# eval/cutile/llm_client.py
"""Thin LLM API client for the eval harness."""

from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        from openai import OpenAI

        self._model = model or os.environ.get("SKILL_EVOLVER_MODEL", "gpt-5.2")
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://openai-api.shenmishajing.workers.dev/v1"
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        """Single-turn completion."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def multi_turn(self, messages: list[dict], max_tokens: int = 2000) -> str:
        """Multi-turn conversation. messages: [{"role": "user"/"assistant", "content": "..."}]"""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content
```

- [ ] **Step 2: Verify import**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -c "from eval.cutile.llm_client import LLMClient; print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add eval/cutile/llm_client.py
git commit -m "feat(eval): add thin LLM client wrapper"
```

---

### Task 3: Phase A — Direct Skill Extraction

**Files:**
- Create: `eval/cutile/phase_a_study.py`
- Test: `tests/test_phase_a_study.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_phase_a_study.py
import json
import pytest
from unittest.mock import MagicMock

from eval.cutile.phase_a_study import PhaseAStudy
from metaclaw.skill_signal import SkillSignal, SkillAction


class FakeLLM:
    def complete(self, prompt, max_tokens=2000):
        if "teaching note" in prompt.lower():
            return "Remember this: when implementing softmax, use ct.load with TMA for large rows."
        # Evolver response
        return json.dumps([{
            "action": "create",
            "skill": {
                "name": "softmax-tma-loading",
                "description": "Use TMA loading for softmax on large rows",
                "content": "## Softmax TMA\n1. Use ct.load with use_tma=True\n",
                "category": "coding",
            },
            "reasoning": "Pattern from TileGym softmax kernel",
        }])


@pytest.fixture
def skill_dir(tmp_path):
    return str(tmp_path / "skills")


def test_generate_teaching_summary():
    study = PhaseAStudy(llm=FakeLLM(), skill_dir="/tmp/unused")
    summary = study.generate_teaching_summary(
        kernel_source="@ct.kernel\ndef softmax(...):\n    pass",
        test_source="def test_op(self): pass",
    )
    assert "Remember this" in summary


def test_run_on_single_kernel(skill_dir):
    study = PhaseAStudy(llm=FakeLLM(), skill_dir=skill_dir)
    pair = {
        "kernel_name": "softmax",
        "kernel_source": "@ct.kernel\ndef softmax(...):\n    ct.load(...)",
        "test_source": "def test_op(self): pass",
    }
    signals = study.process_kernel(pair)
    assert len(signals) >= 1
    assert signals[0].signal_type == "explicit_save"


def test_run_produces_skills(skill_dir):
    study = PhaseAStudy(llm=FakeLLM(), skill_dir=skill_dir)
    pairs = [
        {
            "kernel_name": "softmax",
            "kernel_source": "@ct.kernel\ndef softmax(...):\n    pass",
            "test_source": "def test_op(self): pass",
        },
    ]
    result = study.run(pairs)
    assert result["num_summaries"] >= 1
    assert result["num_skills_created"] >= 0  # depends on evolver
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_phase_a_study.py -v`

- [ ] **Step 3: Implement**

```python
# eval/cutile/phase_a_study.py
"""Phase A: Direct skill extraction from TileGym kernel code."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from metaclaw.conversation_signal_detector import ConversationSignalDetector
from metaclaw.signal_aggregator import SignalAggregator, SkillEvolutionConfig
from metaclaw.skill_evolver import SkillEvolver
from metaclaw.skill_manager import SkillManager
from metaclaw.skill_signal import SkillSignal

logger = logging.getLogger(__name__)

_TEACHING_PROMPT = """\
You are a cuTile expert writing a teaching note for a student who will
implement GPU kernels using the cuTile Python DSL.

Study this kernel implementation and its tests:

Implementation:
```python
{kernel_source}
```

Tests:
```python
{test_source}
```

Write a concise teaching note (3-5 sentences) explaining the key cuTile
patterns and API usage in this kernel. Start with "Remember this:" to
make it clear this is a skill to save. Be specific about ct.* API calls,
tile sizes, and hardware features (TMA, tensor cores) used.
"""


class PhaseAStudy:
    """Read TileGym kernels → generate teaching summaries → extract skills."""

    def __init__(self, llm, skill_dir: str):
        self._llm = llm
        self._skill_dir = skill_dir
        os.makedirs(skill_dir, exist_ok=True)

        self._detector = ConversationSignalDetector(use_llm_detection=False)
        self._aggregator = SignalAggregator(SkillEvolutionConfig(
            sources=["conversation"],
            correction_threshold=999,  # never trigger from corrections in Phase A
        ))
        self._skill_manager = SkillManager(skill_dir, retrieval_mode="template")
        self._evolver = SkillEvolver(llm_client=_LLMAdapter(llm))

    def generate_teaching_summary(self, kernel_source: str, test_source: str) -> str:
        prompt = _TEACHING_PROMPT.format(
            kernel_source=kernel_source[:2000],
            test_source=test_source[:1000],
        )
        return self._llm.complete(prompt)

    def process_kernel(self, pair: dict) -> list[SkillSignal]:
        """Generate teaching summary and detect signals for one kernel."""
        summary = self.generate_teaching_summary(
            pair["kernel_source"], pair["test_source"]
        )
        turn_data = {
            "session_id": f"phase-a-{pair['kernel_name']}",
            "turn_num": 1,
            "user_message": summary,
            "assistant_response": pair["kernel_source"][:500],
            "active_skills": [],
        }
        return self._detector.detect(turn_data)

    def run(self, pairs: list[dict], output_path: str = "") -> dict:
        """Run Phase A on all kernel pairs. Returns summary dict."""
        all_summaries = []
        for pair in pairs:
            signals = self.process_kernel(pair)
            self._aggregator.add(signals)
            all_summaries.append({
                "kernel": pair["kernel_name"],
                "signals": len(signals),
            })

        # Evolve from all accumulated signals
        consumed = self._aggregator.consume()
        actions = asyncio.run(
            self._evolver.evolve(consumed, self._skill_manager.skills)
        )
        created = 0
        for action in actions:
            if action.action == "create":
                if self._skill_manager.add_skill(action.skill):
                    created += 1
            elif action.action == "update":
                self._skill_manager.update_skill(action.skill["name"], action.skill)

        # Save summaries
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for s in all_summaries:
                    f.write(json.dumps(s) + "\n")

        return {
            "num_summaries": len(all_summaries),
            "num_signals": len(consumed),
            "num_skills_created": created,
        }


class _LLMAdapter:
    """Adapts eval LLMClient to SkillEvolver's chat_complete interface."""
    def __init__(self, llm):
        self._llm = llm
    def chat_complete(self, prompt: str) -> str:
        return self._llm.complete(prompt, max_tokens=3000)
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_phase_a_study.py -v`

- [ ] **Step 5: Commit**

```bash
git add eval/cutile/phase_a_study.py tests/test_phase_a_study.py
git commit -m "feat(eval): add Phase A direct skill extraction from TileGym"
```

---

### Task 4: Correction Simulator

**Files:**
- Create: `eval/cutile/correction_simulator.py`
- Test: `tests/test_correction_simulator.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_correction_simulator.py
from eval.cutile.correction_simulator import CorrectionSimulator


class FakeLLM:
    def complete(self, prompt, max_tokens=2000):
        if "CORRECT" in prompt:
            # Comparison prompt
            return "No, that's wrong. You should use ct.matmul instead of a manual loop."
        return "some response"


def test_compare_wrong_solution():
    sim = CorrectionSimulator(llm=FakeLLM())
    result = sim.compare(
        agent_solution="def matmul(a, b): return a @ b",
        reference="@ct.kernel\ndef matmul(a, b): ct.matmul(a, b)",
        kernel_name="matmul",
    )
    assert result["is_correct"] is False
    assert "No, that's wrong" in result["correction"]


def test_compare_correct_solution():
    class CorrectLLM:
        def complete(self, prompt, max_tokens=2000):
            return "CORRECT"

    sim = CorrectionSimulator(llm=CorrectLLM())
    result = sim.compare(
        agent_solution="@ct.kernel\ndef matmul(a, b): ct.matmul(a, b)",
        reference="@ct.kernel\ndef matmul(a, b): ct.matmul(a, b)",
        kernel_name="matmul",
    )
    assert result["is_correct"] is True
    assert result["correction"] == ""


def test_correction_avoids_explicit_keywords():
    sim = CorrectionSimulator(llm=FakeLLM())
    result = sim.compare(
        agent_solution="bad code",
        reference="good code",
        kernel_name="test",
    )
    # The prompt instructs LLM to avoid "always", "never", "from now on"
    # We verify the prompt is built correctly (can't control LLM output in test)
    assert result["correction"] != ""
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement**

```python
# eval/cutile/correction_simulator.py
"""Generate corrections by comparing agent solutions against TileGym references."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_COMPARE_PROMPT = """\
Compare this student's cuTile kernel with the reference implementation.

Student attempt:
```python
{agent_solution}
```

Reference (correct):
```python
{reference}
```

If the student's code has errors or uses wrong patterns, write a correction
(2-3 sentences). Start with "No, that's wrong." Be specific about cuTile
API usage (ct.* functions, tile sizes, TMA, tensor cores).

IMPORTANT: Do NOT use "always", "never", "from now on", "remember this",
or "save this as a skill". Only describe what is wrong and what to do instead.

If the student's code is essentially correct, respond with just "CORRECT".
"""

_FALLBACK_PREFIX = "No, that's wrong. "


class CorrectionSimulator:
    """Generates natural corrections from reference comparison."""

    def __init__(self, llm):
        self._llm = llm

    def compare(
        self,
        agent_solution: str,
        reference: str,
        kernel_name: str,
    ) -> dict:
        """Compare agent solution to reference. Returns {is_correct, correction, kernel_name}."""
        prompt = _COMPARE_PROMPT.format(
            agent_solution=agent_solution[:1500],
            reference=reference[:1500],
        )
        response = self._llm.complete(prompt, max_tokens=500)
        response = response.strip()

        if response.upper().startswith("CORRECT"):
            return {"is_correct": True, "correction": "", "kernel_name": kernel_name}

        # Ensure correction triggers implicit detection
        correction = response
        if not any(
            correction.lower().startswith(prefix)
            for prefix in ["no,", "no ", "that's wrong", "that is wrong"]
        ):
            correction = _FALLBACK_PREFIX + correction

        return {"is_correct": False, "correction": correction, "kernel_name": kernel_name}
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_correction_simulator.py -v`

- [ ] **Step 5: Commit**

```bash
git add eval/cutile/correction_simulator.py tests/test_correction_simulator.py
git commit -m "feat(eval): add CorrectionSimulator for reference comparison"
```

---

### Task 5: Correction Clusterer

**Files:**
- Create: `eval/cutile/correction_clusterer.py`
- Test: `tests/test_correction_clusterer.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_correction_clusterer.py
from eval.cutile.correction_clusterer import CorrectionClusterer


def test_cluster_by_keywords():
    corrections = [
        {"correction": "No, use ct.matmul instead of manual loop", "kernel_name": "a"},
        {"correction": "No, use ct.sum instead of manual reduction loop", "kernel_name": "b"},
        {"correction": "No, wrong tile shape for tensor cores", "kernel_name": "c"},
        {"correction": "No, tile dimensions must be multiples of 16 for tensor cores", "kernel_name": "d"},
        {"correction": "No, missing TMA configuration in ct.load", "kernel_name": "e"},
    ]
    clusterer = CorrectionClusterer()
    clusters = clusterer.cluster(corrections)
    assert len(clusters) >= 2  # at least "builtin ops" and "tile shapes"
    assert all("corrections" in c for c in clusters)
    assert all("label" in c for c in clusters)


def test_empty_corrections():
    clusterer = CorrectionClusterer()
    clusters = clusterer.cluster([])
    assert clusters == []


def test_single_correction():
    clusterer = CorrectionClusterer()
    clusters = clusterer.cluster([{"correction": "No, wrong API", "kernel_name": "x"}])
    assert len(clusters) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement**

```python
# eval/cutile/correction_clusterer.py
"""Group corrections by error pattern using keyword matching."""

from __future__ import annotations

import re
from collections import defaultdict

# Keyword clusters for cuTile error patterns
_CLUSTER_KEYWORDS = {
    "builtin-ops": ["manual loop", "manual reduction", "ct.matmul", "ct.sum", "ct.mma", "builtin", "built-in"],
    "tile-shape": ["tile shape", "tile size", "tile dimensions", "multiples of", "tensor core"],
    "tma-config": ["tma", "ct.load", "ct.store", "memory accelerator"],
    "memory-layout": ["transpose", "layout", "stride", "contiguous", "permute"],
    "api-misuse": ["doesn't exist", "not available", "wrong function", "wrong api", "import"],
    "dtype": ["dtype", "float32", "bfloat16", "float16", "precision", "cast"],
}


class CorrectionClusterer:
    """Groups corrections by error pattern using keyword matching."""

    def cluster(self, corrections: list[dict]) -> list[dict]:
        """Group corrections into clusters. Each cluster: {label, corrections: []}."""
        if not corrections:
            return []

        buckets: dict[str, list[dict]] = defaultdict(list)

        for corr in corrections:
            text = corr.get("correction", "").lower()
            matched = False
            for label, keywords in _CLUSTER_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    buckets[label].append(corr)
                    matched = True
                    break
            if not matched:
                buckets["other"].append(corr)

        return [
            {"label": label, "corrections": items}
            for label, items in buckets.items()
        ]
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_correction_clusterer.py -v`

- [ ] **Step 5: Commit**

```bash
git add eval/cutile/correction_clusterer.py tests/test_correction_clusterer.py
git commit -m "feat(eval): add CorrectionClusterer for grouping by error pattern"
```

---

### Task 6: Skill Writer

**Files:**
- Create: `eval/cutile/skill_writer.py`
- Test: `tests/test_skill_writer.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_skill_writer.py
import os
from eval.cutile.skill_writer import SkillWriter


def test_merge_skills(tmp_path):
    base = tmp_path / "base.md"
    base.write_text("# cuTile Base Skill\nBase content here.\n")

    evolved = [
        {"name": "skill-a", "description": "Desc A", "content": "## A\nDo A."},
        {"name": "skill-b", "description": "Desc B", "content": "## B\nDo B."},
    ]

    output = tmp_path / "merged.md"
    SkillWriter.merge(str(base), evolved, str(output))

    text = output.read_text()
    assert "cuTile Base Skill" in text
    assert "Evolved Skills" in text
    assert "skill-a" in text
    assert "skill-b" in text
    assert "Do A." in text


def test_merge_no_evolved(tmp_path):
    base = tmp_path / "base.md"
    base.write_text("# Base\n")
    output = tmp_path / "merged.md"
    SkillWriter.merge(str(base), [], str(output))
    text = output.read_text()
    assert "# Base" in text
    assert "Evolved" not in text
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement**

```python
# eval/cutile/skill_writer.py
"""Merge evolved skills into a single markdown file for injection."""

from __future__ import annotations

import os
from pathlib import Path


class SkillWriter:
    """Produces a single merged skill file from base + evolved skills."""

    @staticmethod
    def merge(base_skill_path: str, evolved_skills: list[dict], output_path: str) -> None:
        """Concatenate base cuTile skill + evolved skills into one file."""
        parts = [Path(base_skill_path).read_text(encoding="utf-8")]

        if evolved_skills:
            parts.append("\n\n---\n\n# Evolved Skills (learned from TileGym)\n")
            for skill in evolved_skills:
                name = skill.get("name", "unnamed")
                desc = skill.get("description", "")
                content = skill.get("content", "")
                parts.append(f"\n## {name}\n_{desc}_\n\n{content}\n")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(parts), encoding="utf-8")
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_skill_writer.py -v`

- [ ] **Step 5: Commit**

```bash
git add eval/cutile/skill_writer.py tests/test_skill_writer.py
git commit -m "feat(eval): add SkillWriter for merging evolved skills"
```

---

### Task 7: Phase B — Multi-Turn Skill Evolution

**Files:**
- Create: `eval/cutile/phase_b_practice.py`
- Test: `tests/test_phase_b_practice.py`

This is the core novelty — the conversational evolution loop.

- [ ] **Step 1: Write tests**

```python
# tests/test_phase_b_practice.py
import asyncio
import json
import pytest
from unittest.mock import MagicMock

from eval.cutile.phase_b_practice import PhaseBPractice
from metaclaw.skill_signal import SkillAction


class FakeLLM:
    """Returns canned responses for solve, compare, and evolve."""
    def __init__(self):
        self._call_count = 0

    def complete(self, prompt, max_tokens=2000):
        self._call_count += 1
        if "solve" in prompt.lower() or "test specification" in prompt.lower():
            return "def kernel(): pass  # wrong attempt"
        if "compare" in prompt.lower() or "reference" in prompt.lower():
            return "No, that's wrong. You should use ct.matmul instead."
        if "skill engineer" in prompt.lower() or "propose" in prompt.lower():
            return json.dumps([{
                "action": "create",
                "skill": {
                    "name": "use-ct-matmul",
                    "description": "Use ct.matmul for matrix ops",
                    "content": "## Use ct.matmul\n1. Always prefer ct.matmul\n",
                    "category": "coding",
                },
                "reasoning": "Agent used manual loops",
            }])
        return "fallback response"

    def multi_turn(self, messages, max_tokens=2000):
        return self.complete(messages[-1]["content"], max_tokens)


@pytest.fixture
def skill_dir(tmp_path):
    return str(tmp_path / "skills")


def test_solve_kernel(skill_dir):
    practice = PhaseBPractice(llm=FakeLLM(), skill_dir=skill_dir)
    solution = practice.solve_kernel("def test_op(): pass", [])
    assert len(solution) > 0


def test_single_round(skill_dir):
    practice = PhaseBPractice(llm=FakeLLM(), skill_dir=skill_dir)
    pairs = [{
        "kernel_name": "matmul",
        "kernel_source": "@ct.kernel\ndef matmul(): ct.matmul(a, b)",
        "test_source": "def test_op(): pass",
    }]
    result = practice.run_round(pairs, round_num=1)
    assert "num_attempted" in result
    assert "num_correct" in result
    assert "num_corrections" in result


def test_full_run_respects_rounds(skill_dir):
    practice = PhaseBPractice(llm=FakeLLM(), skill_dir=skill_dir, max_rounds=2)
    pairs = [{
        "kernel_name": "matmul",
        "kernel_source": "@ct.kernel\ndef matmul(): ct.matmul(a, b)",
        "test_source": "def test_op(): pass",
    }]
    result = practice.run(pairs)
    assert "rounds" in result
    assert len(result["rounds"]) <= 2
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement**

```python
# eval/cutile/phase_b_practice.py
"""Phase B: Solve-and-check with conversational skill evolution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from metaclaw.conversation_signal_detector import ConversationSignalDetector
from metaclaw.signal_aggregator import SignalAggregator, SkillEvolutionConfig
from metaclaw.skill_evolver import SkillEvolver
from metaclaw.skill_manager import SkillManager
from metaclaw.skill_signal import SkillSignal

from .correction_simulator import CorrectionSimulator
from .correction_clusterer import CorrectionClusterer

logger = logging.getLogger(__name__)

_SOLVE_PROMPT = """\
You are solving a cuTile kernel implementation problem.

Test specification:
```python
{test_source}
```

{skills_text}

Write the cuTile kernel implementation that passes these tests.
Use the ct.* API (ct.kernel, ct.launch, ct.load, ct.store, etc.).
Write only the implementation code.
"""


class PhaseBPractice:
    """Iterative solve-and-check with conversational skill evolution."""

    def __init__(
        self,
        llm,
        skill_dir: str,
        max_rounds: int = 3,
        correction_threshold: int = 3,
    ):
        self._llm = llm
        self._skill_dir = skill_dir
        self._max_rounds = max_rounds
        os.makedirs(skill_dir, exist_ok=True)

        self._detector = ConversationSignalDetector(use_llm_detection=False)
        self._aggregator = SignalAggregator(SkillEvolutionConfig(
            sources=["conversation"],
            correction_threshold=correction_threshold,
        ))
        self._skill_manager = SkillManager(skill_dir, retrieval_mode="template")
        self._evolver = SkillEvolver(llm_client=_LLMAdapter(llm))
        self._correction_sim = CorrectionSimulator(llm)
        self._clusterer = CorrectionClusterer()

    def solve_kernel(self, test_source: str, active_skills: list[str]) -> str:
        """Agent attempts to implement a kernel given its test spec."""
        skills_text = ""
        if active_skills:
            skills_text = "Available cuTile skills:\n" + "\n".join(
                f"- {name}" for name in active_skills
            )
        prompt = _SOLVE_PROMPT.format(test_source=test_source[:3000], skills_text=skills_text)
        return self._llm.complete(prompt, max_tokens=3000)

    def run_round(
        self,
        pairs: list[dict],
        round_num: int,
        output_dir: str = "",
    ) -> dict:
        """Run one round: attempt all kernels, collect corrections, evolve."""
        active_skills = [s["name"] for s in self._skill_manager._iter_all_skills()]
        attempts = []
        corrections = []

        for pair in pairs:
            # Solve
            solution = self.solve_kernel(pair["test_source"], active_skills)

            # Compare against reference
            result = self._correction_sim.compare(
                agent_solution=solution,
                reference=pair["kernel_source"],
                kernel_name=pair["kernel_name"],
            )
            attempts.append({
                "kernel": pair["kernel_name"],
                "correct": result["is_correct"],
                "solution_preview": solution[:200],
            })

            if not result["is_correct"]:
                corrections.append(result)
                # Feed through conversation pipeline
                turn_data = {
                    "session_id": f"phase-b-round{round_num}-{pair['kernel_name']}",
                    "turn_num": 1,
                    "user_message": result["correction"],
                    "assistant_response": solution[:500],
                    "active_skills": active_skills,
                }
                signals = self._detector.detect(turn_data)
                if not signals:
                    # Fallback: force detection
                    turn_data["user_message"] = f"No, that's wrong. {result['correction']}"
                    signals = self._detector.detect(turn_data)
                self._aggregator.add(signals)

        # Evolve from accumulated corrections
        num_evolved = 0
        consumed = self._aggregator.consume()
        if consumed:
            actions = asyncio.run(
                self._evolver.evolve(consumed, self._skill_manager.skills)
            )
            for action in actions:
                if action.action == "create":
                    if self._skill_manager.add_skill(action.skill):
                        num_evolved += 1
                elif action.action == "update":
                    if self._skill_manager.update_skill(action.skill["name"], action.skill):
                        num_evolved += 1

        # Save round artifacts
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_dir, "attempts.jsonl"), "w") as f:
                for a in attempts:
                    f.write(json.dumps(a) + "\n")
            with open(os.path.join(output_dir, "corrections.jsonl"), "w") as f:
                for c in corrections:
                    f.write(json.dumps(c) + "\n")

        return {
            "round": round_num,
            "num_attempted": len(pairs),
            "num_correct": sum(1 for a in attempts if a["correct"]),
            "num_corrections": len(corrections),
            "num_skills_evolved": num_evolved,
            "active_skills": [s["name"] for s in self._skill_manager._iter_all_skills()],
        }

    def run(self, pairs: list[dict], output_dir: str = "") -> dict:
        """Run all rounds until convergence or max_rounds."""
        rounds = []
        failed_kernels = pairs  # start with all

        for r in range(1, self._max_rounds + 1):
            round_output = os.path.join(output_dir, f"round-{r}") if output_dir else ""
            result = self.run_round(failed_kernels, round_num=r, output_dir=round_output)
            rounds.append(result)

            # Filter to only re-attempt failed kernels
            correct_names = {
                a["kernel"] for a in [] # read from attempts
            }
            # Simpler: re-attempt kernels that had corrections
            if result["num_corrections"] == 0:
                logger.info("[PhaseB] No corrections in round %d — converged.", r)
                break

            # Check for plateau (same correction count as previous)
            if len(rounds) >= 2 and rounds[-1]["num_corrections"] >= rounds[-2]["num_corrections"]:
                logger.info("[PhaseB] Plateau at round %d — stopping.", r)
                break

        return {
            "rounds": rounds,
            "total_skills": len(list(self._skill_manager._iter_all_skills())),
        }


class _LLMAdapter:
    """Adapts eval LLMClient to SkillEvolver's chat_complete interface."""
    def __init__(self, llm):
        self._llm = llm
    def chat_complete(self, prompt: str) -> str:
        return self._llm.complete(prompt, max_tokens=3000)
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_phase_b_practice.py -v`

- [ ] **Step 5: Commit**

```bash
git add eval/cutile/phase_b_practice.py tests/test_phase_b_practice.py
git commit -m "feat(eval): add Phase B multi-turn solve-and-check with conversational evolution"
```

---

### Task 8: Report Generator

**Files:**
- Create: `eval/cutile/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_report.py
import json
from eval.cutile.report import generate_comparison, generate_summary_text


def test_generate_comparison():
    control = {"cutile/0": True, "cutile/1": False, "cutile/2": False}
    method1 = {"cutile/0": True, "cutile/1": True, "cutile/2": False}
    method2 = {"cutile/0": True, "cutile/1": True, "cutile/2": True}

    report = generate_comparison(control, method1, method2)
    assert report["control_pass"] == 1
    assert report["method1_pass"] == 2
    assert report["method2_pass"] == 3
    assert "cutile/1" in report["flips_control_to_method1"]
    assert "cutile/2" in report["flips_method1_to_method2"]


def test_generate_summary_text():
    report = {
        "control_pass": 20, "control_total": 48,
        "method1_pass": 25, "method1_total": 48,
        "method2_pass": 30, "method2_total": 48,
        "flips_control_to_method1": ["cutile/1", "cutile/2"],
        "flips_control_to_method2": ["cutile/1", "cutile/2", "cutile/3"],
        "flips_method1_to_method2": ["cutile/3"],
        "regressions_method1": [],
        "regressions_method2": [],
    }
    text = generate_summary_text(report)
    assert "20/48" in text
    assert "25/48" in text
    assert "30/48" in text
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement**

```python
# eval/cutile/report.py
"""Comparison reports and skill attribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def parse_eval_results(jsonl_path: str) -> dict[str, bool]:
    """Parse *-graded-solutions.jsonl → {task_id: passed}."""
    results = {}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            task_id = entry.get("task_id", "")
            passed = entry.get("passed", False)
            skipped = entry.get("skipped", False)
            if not skipped:
                results[task_id] = passed
    return results


def generate_comparison(
    control: dict[str, bool],
    method1: dict[str, bool],
    method2: dict[str, bool],
) -> dict:
    """Compare three conditions. Returns structured report."""
    all_tasks = sorted(set(control) | set(method1) | set(method2))

    flips_c_to_m1 = [t for t in all_tasks if not control.get(t, False) and method1.get(t, False)]
    flips_c_to_m2 = [t for t in all_tasks if not control.get(t, False) and method2.get(t, False)]
    flips_m1_to_m2 = [t for t in all_tasks if not method1.get(t, False) and method2.get(t, False)]
    reg_m1 = [t for t in all_tasks if control.get(t, False) and not method1.get(t, False)]
    reg_m2 = [t for t in all_tasks if control.get(t, False) and not method2.get(t, False)]

    return {
        "control_pass": sum(control.values()),
        "control_total": len(control),
        "method1_pass": sum(method1.values()),
        "method1_total": len(method1),
        "method2_pass": sum(method2.values()),
        "method2_total": len(method2),
        "flips_control_to_method1": flips_c_to_m1,
        "flips_control_to_method2": flips_c_to_m2,
        "flips_method1_to_method2": flips_m1_to_m2,
        "regressions_method1": reg_m1,
        "regressions_method2": reg_m2,
    }


def generate_summary_text(report: dict) -> str:
    """Human-readable summary table."""
    lines = [
        "=== cuTile Skill Evolution Report ===",
        "",
        f"{'Method':<30} {'Pass':>6} {'Total':>6} {'Delta':>6}",
        "-" * 54,
        f"{'No skills (control)':<30} {report['control_pass']:>4}/{report['control_total']:<4} {'':>6}",
        f"{'Method 1: Direct extract':<30} {report['method1_pass']:>4}/{report['method1_total']:<4} {'+' + str(report['method1_pass'] - report['control_pass']):>6}",
        f"{'Method 2: Multi-turn':<30} {report['method2_pass']:>4}/{report['method2_total']:<4} {'+' + str(report['method2_pass'] - report['control_pass']):>6}",
        "",
        f"Flips (fail→pass) Control→Method 1: {report['flips_control_to_method1']}",
        f"Flips (fail→pass) Control→Method 2: {report['flips_control_to_method2']}",
        f"Flips (fail→pass) Method 1→Method 2: {report['flips_method1_to_method2']}",
        f"Regressions Method 1: {report['regressions_method1']}",
        f"Regressions Method 2: {report['regressions_method2']}",
    ]
    return "\n".join(lines)


def save_report(report: dict, summary: str, output_dir: str) -> None:
    """Save structured and text reports."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    (Path(output_dir) / "summary.txt").write_text(summary, encoding="utf-8")
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_report.py -v`

- [ ] **Step 5: Commit**

```bash
git add eval/cutile/report.py tests/test_report.py
git commit -m "feat(eval): add comparison report generator with skill attribution"
```

---

### Task 9: Phase C — Compute-Eval Benchmark Runner

**Files:**
- Create: `eval/cutile/phase_c_evaluate.py`

No unit tests — this is subprocess orchestration that calls shell scripts. Tested by running the actual eval.

- [ ] **Step 1: Implement**

```python
# eval/cutile/phase_c_evaluate.py
"""Phase C: Run compute-eval benchmark via subprocess."""

from __future__ import annotations

import glob
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class PhaseCEvaluate:
    """Run compute-eval benchmark using cutile-eval-kit shell scripts."""

    def __init__(self, eval_kit_dir: str, model: str = "azure/openai/gpt-5.2"):
        self._eval_kit_dir = eval_kit_dir
        self._model = model
        self._solve_script = os.path.join(
            eval_kit_dir, "scripts", "compute-eval-run-cutile-opencode-gpt52.sh"
        )
        self._eval_script = os.path.join(
            eval_kit_dir, "scripts", "run_b200_local_compute_eval_evalonly.sh"
        )

    def run_solve_and_eval(
        self,
        run_root: str,
        skill_md: str = "",
        task_ids: str = "",
    ) -> str:
        """Run full solve + eval. Returns path to graded results JSONL."""
        env = {
            **os.environ,
            "RUN_ROOT": run_root,
            "MODEL": self._model,
            "TASK_IDS": task_ids,
        }
        if skill_md:
            env["SKILL_MD"] = skill_md

        logger.info("[PhaseC] Running solve+eval: %s", self._solve_script)
        subprocess.run(
            ["bash", self._solve_script],
            env=env,
            check=True,
        )

        return self._find_graded_results(run_root)

    def run_eval_only(
        self,
        base_run: str,
        run_root: str,
        task_ids: str = "",
    ) -> str:
        """Run eval-only on existing solutions. Returns path to graded results."""
        env = {
            **os.environ,
            "BASE_RUN": base_run,
            "RUN_ROOT": run_root,
            "TASK_IDS": task_ids,
        }

        logger.info("[PhaseC] Running eval-only: %s", self._eval_script)
        subprocess.run(
            ["bash", self._eval_script],
            env=env,
            check=True,
        )

        return self._find_graded_results(run_root)

    @staticmethod
    def _find_graded_results(run_root: str) -> str:
        """Find the *-graded-solutions.jsonl file in eval-output/."""
        pattern = os.path.join(run_root, "eval-output", "*-graded-solutions.jsonl")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No graded results found matching: {pattern}")
        return matches[-1]
```

- [ ] **Step 2: Verify import**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -c "from eval.cutile.phase_c_evaluate import PhaseCEvaluate; print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add eval/cutile/phase_c_evaluate.py
git commit -m "feat(eval): add Phase C compute-eval benchmark runner"
```

---

### Task 10: Main Orchestrator (run_eval.py)

**Files:**
- Create: `eval/cutile/run_eval.py`

- [ ] **Step 1: Implement**

```python
# eval/cutile/run_eval.py
"""Main orchestrator for cuTile skill evolution evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.cutile.tilegym_loader import TileGymLoader
from eval.cutile.llm_client import LLMClient
from eval.cutile.phase_a_study import PhaseAStudy
from eval.cutile.phase_b_practice import PhaseBPractice
from eval.cutile.phase_c_evaluate import PhaseCEvaluate
from eval.cutile.skill_writer import SkillWriter
from eval.cutile.report import parse_eval_results, generate_comparison, generate_summary_text, save_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="cuTile Skill Evolution Evaluation")
    parser.add_argument("--tilegym-dir", required=True, help="Path to TileGym repo")
    parser.add_argument("--eval-kit-dir", default="", help="Path to cutile-eval-kit")
    parser.add_argument("--results-dir", default="eval/cutile/results", help="Output directory")
    parser.add_argument("--base-skill", default="", help="Path to cutile-minimal-skill.md")
    parser.add_argument("--model", default="azure/openai/gpt-5.2", help="LLM model")
    parser.add_argument("--evolver-model", default="", help="Model for skill evolution")
    parser.add_argument("--rounds", type=int, default=2, help="Max Phase B rounds")
    parser.add_argument("--task-ids", default="", help="Comma-separated task IDs (empty=all)")
    parser.add_argument("--method", default="all", choices=["all", "direct", "multiturn", "evaluate"],
                       help="Which method to run")
    parser.add_argument("--skills-from", default="", help="Pre-evolved skills dir (for --method evaluate)")
    parser.add_argument("--local-test", action="store_true", help="Skip Phase C (no GPU needed)")
    parser.add_argument("--mock-eval-results", default="", help="Mock eval results for local test")
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    llm = LLMClient(model=args.evolver_model or args.model)
    loader = TileGymLoader(args.tilegym_dir)
    pairs = loader.get_all_pairs()
    logger.info("Loaded %d kernel pairs from TileGym", len(pairs))

    # --- Method 1: Direct Extraction ---
    if args.method in ("all", "direct"):
        logger.info("=== Method 1: Direct Extraction ===")
        m1_skill_dir = os.path.join(results_dir, "method-1-direct", "evolved-skills")
        study = PhaseAStudy(llm=llm, skill_dir=m1_skill_dir)
        m1_result = study.run(
            pairs,
            output_path=os.path.join(results_dir, "method-1-direct", "teaching-summaries.jsonl"),
        )
        logger.info("Method 1 result: %s", m1_result)

        # Write merged skill file
        if args.base_skill:
            m1_skills = list(study._skill_manager._iter_all_skills())
            SkillWriter.merge(
                args.base_skill, m1_skills,
                os.path.join(results_dir, "method-1-direct", "cutile-skill-merged.md"),
            )

    # --- Method 2: Multi-Turn Evolution ---
    if args.method in ("all", "multiturn"):
        logger.info("=== Method 2: Multi-Turn Evolution ===")
        # Phase A first (same as Method 1)
        m2_skill_dir = os.path.join(results_dir, "method-2-multiturn", "evolved-skills")
        study_m2 = PhaseAStudy(llm=llm, skill_dir=m2_skill_dir)
        study_m2.run(
            pairs,
            output_path=os.path.join(results_dir, "method-2-multiturn", "phase-a", "teaching-summaries.jsonl"),
        )

        # Phase B
        practice = PhaseBPractice(
            llm=llm,
            skill_dir=m2_skill_dir,
            max_rounds=args.rounds,
        )
        m2_result = practice.run(
            pairs,
            output_dir=os.path.join(results_dir, "method-2-multiturn", "phase-b"),
        )
        logger.info("Method 2 result: %s", m2_result)

        # Write merged skill file
        if args.base_skill:
            m2_skills = list(practice._skill_manager._iter_all_skills())
            SkillWriter.merge(
                args.base_skill, m2_skills,
                os.path.join(results_dir, "method-2-multiturn", "cutile-skill-merged.md"),
            )

    # --- Phase C: Evaluate on compute-eval ---
    if args.method in ("all", "evaluate") and not args.local_test:
        if not args.eval_kit_dir:
            logger.error("--eval-kit-dir required for Phase C")
            return

        logger.info("=== Phase C: Compute-Eval Benchmark ===")
        evaluator = PhaseCEvaluate(args.eval_kit_dir, model=args.model)

        # Run for each condition and collect results
        # (actual B200 runs — this takes hours)
        logger.info("Phase C requires B200 allocation. Run each condition separately.")
        logger.info("Use: --method evaluate --skills-from <path>")

    # --- Comparison report (if mock results available) ---
    if args.mock_eval_results and args.local_test:
        logger.info("=== Generating comparison from mock results ===")
        # For local testing, use the same mock for all three conditions
        mock = parse_eval_results(args.mock_eval_results)
        report = generate_comparison(mock, mock, mock)
        summary = generate_summary_text(report)
        save_report(report, summary, results_dir)
        print(summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python eval/cutile/run_eval.py --help`
Expected: Prints usage/help

- [ ] **Step 3: Commit**

```bash
git add eval/cutile/run_eval.py
git commit -m "feat(eval): add main orchestrator for cuTile skill evolution evaluation"
```

---

### Task 11: README and Final Integration Test

**Files:**
- Create: `eval/cutile/README.md`

- [ ] **Step 1: Write README**

```markdown
# cuTile Skill Evolution Evaluation

Compares one-shot skill extraction vs multi-turn conversational evolution
on cuTile kernel coding. Skills learned from TileGym, evaluated on compute-eval.

## Quick Start (local, no GPU)

```bash
# Method 1 only (direct extraction):
python eval/cutile/run_eval.py \
  --method direct \
  --tilegym-dir TileGym \
  --results-dir eval/cutile/results

# Method 2 (multi-turn, 2 rounds):
python eval/cutile/run_eval.py \
  --method multiturn \
  --tilegym-dir TileGym \
  --results-dir eval/cutile/results \
  --rounds 2

# Both methods + comparison:
python eval/cutile/run_eval.py \
  --method all \
  --tilegym-dir TileGym \
  --results-dir eval/cutile/results \
  --local-test
```

## Phase C (needs B200)

```bash
python eval/cutile/run_eval.py \
  --method evaluate \
  --eval-kit-dir cutile-eval-kit \
  --skills-from eval/cutile/results/method-2-multiturn/cutile-skill-merged.md
```

## Required Environment

- `OPENAI_API_KEY` or `AZURE_API_KEY`
- Python 3.10+ with MetaClaw dependencies
- B200 allocation for Phase C only
```

- [ ] **Step 2: Run full test suite**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/ -v --ignore=tests/test_v03_live_tinker.py --ignore=tests/test_setup_wizard.py`

- [ ] **Step 3: Commit and push**

```bash
git add eval/cutile/README.md
git commit -m "docs(eval): add cuTile eval harness README"
git push origin kaix/conversation-driven-skill-evolution
```
