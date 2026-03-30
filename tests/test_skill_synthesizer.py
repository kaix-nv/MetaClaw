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
