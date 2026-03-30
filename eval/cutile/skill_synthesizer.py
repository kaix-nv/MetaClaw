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
