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
                a["kernel"] for a in []  # read from attempts
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
