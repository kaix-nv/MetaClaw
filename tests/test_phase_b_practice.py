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


def test_solve_kernel(skill_dir, tmp_path):
    practice = PhaseBPractice(llm=FakeLLM(), skill_dir=skill_dir, tilegym_dir=str(tmp_path))
    solution = practice.solve_kernel("def test_op(): pass", [])
    assert len(solution) > 0
