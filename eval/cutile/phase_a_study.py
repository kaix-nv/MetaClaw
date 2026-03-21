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

    def __init__(self, llm, skill_dir: str, base_skill: str = ""):
        self._llm = llm
        self._skill_dir = skill_dir
        os.makedirs(skill_dir, exist_ok=True)

        # Load base skill into SkillManager so evolver sees it and avoids duplication
        if base_skill and os.path.isfile(base_skill):
            self._load_base_skill(skill_dir, base_skill)

        self._detector = ConversationSignalDetector(use_llm_detection=False)
        self._aggregator = SignalAggregator(SkillEvolutionConfig(
            sources=["conversation"],
            correction_threshold=999,  # never trigger from corrections in Phase A
        ))
        self._skill_manager = SkillManager(skill_dir, retrieval_mode="template")
        self._evolver = SkillEvolver(llm_client=_LLMAdapter(llm))

    @staticmethod
    def _load_base_skill(skill_dir: str, base_skill: str) -> None:
        """Copy base skill into the skill dir so SkillManager loads it."""
        base_dir = os.path.join(skill_dir, "cutile-base-skill")
        os.makedirs(base_dir, exist_ok=True)
        dest = os.path.join(base_dir, "SKILL.md")
        if not os.path.exists(dest):
            content = Path(base_skill).read_text(encoding="utf-8")
            # Wrap in frontmatter so SkillManager can parse it
            with open(dest, "w", encoding="utf-8") as f:
                f.write("---\n")
                f.write("name: cutile-base-skill\n")
                f.write("description: Base cuTile Python API reference and patterns\n")
                f.write("category: coding\n")
                f.write("---\n\n")
                f.write(content)

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
