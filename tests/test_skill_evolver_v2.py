import asyncio
import json

import pytest

from metaclaw.skill_evolver import SkillEvolver
from metaclaw.skill_signal import SkillSignal, SkillAction


class FakeLLMClient:
    def __init__(self, response: str):
        self._response = response

    def chat_complete(self, prompt: str) -> str:
        return self._response


def _sig(signal_type="explicit_save", content=None):
    return SkillSignal(
        source="conversation",
        signal_type=signal_type,
        content=content or {"user_message": "test"},
        confidence=0.9,
        timestamp="2026-03-20T10:00:00",
        session_id="s1",
    )


def _skills():
    return {
        "general_skills": [{"name": "existing-skill", "description": "exists"}],
        "task_specific_skills": {},
        "common_mistakes": [],
    }


def test_evolve_creates_skill():
    llm_response = json.dumps([{
        "action": "create",
        "skill": {
            "name": "new-skill",
            "description": "A new skill",
            "content": "# New\nDo this.",
            "category": "general",
        },
        "reasoning": "User explicitly asked",
    }])
    evolver = SkillEvolver(llm_client=FakeLLMClient(llm_response))
    actions = asyncio.run(evolver.evolve([_sig()], _skills()))
    assert len(actions) == 1
    assert isinstance(actions[0], SkillAction)
    assert actions[0].action == "create"
    assert actions[0].skill["name"] == "new-skill"


def test_evolve_empty_signals():
    evolver = SkillEvolver(llm_client=FakeLLMClient("[]"))
    actions = asyncio.run(evolver.evolve([], _skills()))
    assert actions == []


def test_evolve_update_action():
    llm_response = json.dumps([{
        "action": "update",
        "skill": {"name": "existing-skill", "description": "updated", "content": "new", "category": "general"},
        "reasoning": "corrected",
    }])
    evolver = SkillEvolver(llm_client=FakeLLMClient(llm_response))
    actions = asyncio.run(evolver.evolve([_sig("implicit_correction")], _skills()))
    assert actions[0].action == "update"


def test_evolve_link_action():
    llm_response = json.dumps([{
        "action": "link",
        "skill": {"name": "child-skill"},
        "parent_skill": "existing-skill",
        "reasoning": "fits as sub-skill",
    }])
    evolver = SkillEvolver(llm_client=FakeLLMClient(llm_response))
    actions = asyncio.run(evolver.evolve([_sig()], _skills()))
    assert actions[0].action == "link"
    assert actions[0].parent_skill == "existing-skill"


def test_evolve_llm_failure_returns_empty():
    class FailClient:
        def chat_complete(self, prompt):
            raise RuntimeError("LLM down")

    evolver = SkillEvolver(llm_client=FailClient())
    actions = asyncio.run(evolver.evolve([_sig()], _skills()))
    assert actions == []


def test_evolve_passes_existing_skills_to_prompt():
    captured = {}

    class CapturingClient:
        def chat_complete(self, prompt):
            captured["prompt"] = prompt
            return "[]"

    evolver = SkillEvolver(llm_client=CapturingClient())
    asyncio.run(evolver.evolve([_sig()], _skills()))
    assert "existing-skill" in captured["prompt"]
