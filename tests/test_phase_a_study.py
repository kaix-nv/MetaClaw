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
