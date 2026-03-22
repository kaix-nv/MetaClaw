"""Phase B: Solve-and-check with skill evolution from successful fixes only."""

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
from .phase_a_study import PhaseAStudy

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

_SOLVE_WITH_HINT_PROMPT = """\
You are solving a cuTile kernel implementation problem.
Your previous attempt was incorrect. Here is the correct reference implementation:

```python
{reference_code}
```

Study the reference carefully, then re-implement the solution.
Focus on the cuTile patterns the reference uses that you missed.

Test specification:
```python
{test_source}
```

{skills_text}

Write the corrected cuTile kernel implementation.
Use the ct.* API (ct.kernel, ct.launch, ct.load, ct.store, etc.).
Write only the implementation code.
"""

_DIFF_SKILL_PROMPT = """\
Compare these two cuTile kernel implementations for the same problem.
The first attempt FAILED, the second attempt SUCCEEDED after receiving a hint.

FAILED attempt:
```python
{failed_attempt}
```

FIXED attempt (working):
```python
{fixed_attempt}
```

What specific change made it work? Write a teaching note (2-3 sentences)
starting with "Remember this:" that captures the WORKING pattern.
Be specific about cuTile API usage (ct.* functions, tile sizes, TMA, etc.).
"""


class PhaseBPractice:
    """Iterative solve-and-check: only learn from successful fixes."""

    def __init__(
        self,
        llm,
        skill_dir: str,
        max_rounds: int = 3,
        correction_threshold: int = 3,
        base_skill: str = "",
    ):
        self._llm = llm
        self._skill_dir = skill_dir
        self._max_rounds = max_rounds
        os.makedirs(skill_dir, exist_ok=True)

        if base_skill and os.path.isfile(base_skill):
            PhaseAStudy._load_base_skill(skill_dir, base_skill)

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

    def solve_kernel_with_hint(
        self, test_source: str, reference_code: str, active_skills: list[str]
    ) -> str:
        """Agent re-attempts after seeing the reference implementation."""
        skills_text = ""
        if active_skills:
            skills_text = "Available cuTile skills:\n" + "\n".join(
                f"- {name}" for name in active_skills
            )
        prompt = _SOLVE_WITH_HINT_PROMPT.format(
            test_source=test_source[:3000],
            reference_code=reference_code[:3000],
            skills_text=skills_text,
        )
        return self._llm.complete(prompt, max_tokens=3000)

    def generate_diff_skill(self, failed_attempt: str, fixed_attempt: str) -> str:
        """Generate a teaching note from the diff between failed and fixed attempts."""
        prompt = _DIFF_SKILL_PROMPT.format(
            failed_attempt=failed_attempt[:1500],
            fixed_attempt=fixed_attempt[:1500],
        )
        return self._llm.complete(prompt, max_tokens=500)

    def run_round(
        self,
        pairs: list[dict],
        round_num: int,
        output_dir: str = "",
    ) -> dict:
        """Run one round: attempt → correct → re-attempt → learn from fixes."""
        active_skills = [s["name"] for s in self._skill_manager._iter_all_skills()]
        attempts = []
        fixes = []
        still_failing = []

        for pair in pairs:
            # Step 1: First attempt
            first_attempt = self.solve_kernel(pair["test_source"], active_skills)
            result = self._correction_sim.compare(
                agent_solution=first_attempt,
                reference=pair["kernel_source"],
                kernel_name=pair["kernel_name"],
            )

            if result["is_correct"]:
                attempts.append({
                    "kernel": pair["kernel_name"],
                    "status": "passed_first",
                    "solution_preview": first_attempt[:200],
                })
                continue

            # Step 2: Re-attempt with reference code visible
            fixed_attempt = self.solve_kernel_with_hint(
                pair["test_source"], pair["kernel_source"], active_skills
            )
            re_result = self._correction_sim.compare(
                agent_solution=fixed_attempt,
                reference=pair["kernel_source"],
                kernel_name=pair["kernel_name"],
            )

            if re_result["is_correct"]:
                # SUCCESS: Learn from what the fix did right
                diff_message = self.generate_diff_skill(first_attempt, fixed_attempt)

                # Feed the positive teaching note through conversation pipeline
                turn_data = {
                    "session_id": f"phase-b-round{round_num}-{pair['kernel_name']}",
                    "turn_num": 1,
                    "user_message": diff_message,
                    "assistant_response": fixed_attempt[:500],
                    "active_skills": active_skills,
                }
                signals = self._detector.detect(turn_data)
                if not signals:
                    turn_data["user_message"] = f"Remember this: {diff_message}"
                    signals = self._detector.detect(turn_data)
                self._aggregator.add(signals)

                attempts.append({
                    "kernel": pair["kernel_name"],
                    "status": "fixed",
                    "solution_preview": fixed_attempt[:200],
                })
                fixes.append({
                    "kernel": pair["kernel_name"],
                    "initial_error": result["correction"],
                    "diff_skill": diff_message,
                })
            else:
                # Still failing: do NOT learn from this
                attempts.append({
                    "kernel": pair["kernel_name"],
                    "status": "still_failing",
                    "solution_preview": first_attempt[:200],
                })
                still_failing.append(pair)

        # Evolve from successful fixes only
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
            with open(os.path.join(output_dir, "fixes.jsonl"), "w") as f:
                for fix in fixes:
                    f.write(json.dumps(fix) + "\n")

        num_passed_first = sum(1 for a in attempts if a["status"] == "passed_first")
        num_fixed = sum(1 for a in attempts if a["status"] == "fixed")
        num_still_failing = sum(1 for a in attempts if a["status"] == "still_failing")

        logger.info(
            "[PhaseB] Round %d: %d passed_first, %d fixed, %d still_failing, %d skills evolved",
            round_num, num_passed_first, num_fixed, num_still_failing, num_evolved,
        )

        return {
            "round": round_num,
            "num_attempted": len(pairs),
            "num_passed_first": num_passed_first,
            "num_fixed": num_fixed,
            "num_still_failing": num_still_failing,
            "num_skills_evolved": num_evolved,
            "active_skills": [s["name"] for s in self._skill_manager._iter_all_skills()],
            "_still_failing_pairs": still_failing,
        }

    def run(self, pairs: list[dict], output_dir: str = "") -> dict:
        """Run all rounds until convergence or max_rounds."""
        rounds = []
        current_pairs = pairs

        for r in range(1, self._max_rounds + 1):
            round_output = os.path.join(output_dir, f"round-{r}") if output_dir else ""
            result = self.run_round(current_pairs, round_num=r, output_dir=round_output)
            rounds.append(result)

            # Next round: only re-attempt still-failing kernels
            current_pairs = result.pop("_still_failing_pairs", [])

            if not current_pairs:
                logger.info("[PhaseB] All kernels passed or fixed in round %d — converged.", r)
                break

            if result["num_fixed"] == 0:
                logger.info("[PhaseB] No fixes in round %d — plateau.", r)
                break

        return {
            "rounds": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rounds],
            "total_skills": len(list(self._skill_manager._iter_all_skills())),
        }


class _LLMAdapter:
    """Adapts eval LLMClient to SkillEvolver's chat_complete interface."""
    def __init__(self, llm):
        self._llm = llm

    def chat_complete(self, prompt: str) -> str:
        return self._llm.complete(prompt, max_tokens=3000)
