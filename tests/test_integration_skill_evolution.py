"""Integration tests for the full conversation-driven skill evolution pipeline."""

import asyncio
import json
import os

import pytest

from metaclaw.skill_signal import SkillSignal, SkillAction
from metaclaw.signal_aggregator import SignalAggregator, SkillEvolutionConfig
from metaclaw.conversation_signal_detector import ConversationSignalDetector
from metaclaw.feedback_collector import FeedbackCollector
from metaclaw.skill_evolver import SkillEvolver
from metaclaw.skill_manager import SkillManager


class FakeLLM:
    """Returns a canned create-action response."""
    def __init__(self, skill_name="check-logs-first", description="Always check logs before debugging"):
        self._skill_name = skill_name
        self._description = description

    def chat_complete(self, prompt: str) -> str:
        return json.dumps([{
            "action": "create",
            "skill": {
                "name": self._skill_name,
                "description": self._description,
                "content": f"# {self._skill_name}\n\n1. Check logs first\n2. Then reproduce\n",
                "category": "coding",
            },
            "reasoning": "User requested this behavior",
        }])


@pytest.fixture
def skill_dir(tmp_path):
    """Create a skills dir with one pre-existing skill."""
    s = tmp_path / "existing-skill" / "SKILL.md"
    s.parent.mkdir()
    s.write_text(
        "---\nname: existing-skill\ndescription: Pre-existing skill\ncategory: coding\n---\n\n# Existing\nDo existing things.\n"
    )
    return str(tmp_path)


@pytest.fixture
def skill_manager(skill_dir):
    return SkillManager(skill_dir, retrieval_mode="template")


@pytest.fixture
def aggregator():
    return SignalAggregator(SkillEvolutionConfig(
        sources=["conversation", "feedback"],
        correction_threshold=3,
    ))


@pytest.fixture
def detector():
    return ConversationSignalDetector(use_llm_detection=False)


@pytest.fixture
def evolver():
    return SkillEvolver(llm_client=FakeLLM())


@pytest.fixture
def feedback_collector(skill_manager):
    return FeedbackCollector(skill_manager)


# --- Scenario 1: Explicit save flow ---

def test_explicit_save_end_to_end(detector, aggregator, evolver, skill_manager):
    """User says 'from now on...' → signal → evolve → skill on disk."""
    # Step 1: Detect signal from conversation turn
    turn = {
        "session_id": "sess-1",
        "turn_num": 1,
        "user_message": "From now on, always check the logs before debugging anything",
        "assistant_response": "Sure, I'll keep that in mind.",
        "active_skills": [],
    }
    signals = detector.detect(turn)
    assert len(signals) == 1
    assert signals[0].signal_type == "explicit_save"

    # Step 2: Add to aggregator and check trigger
    aggregator.add(signals)
    assert aggregator.should_evolve() is True

    # Step 3: Consume and evolve
    consumed = aggregator.consume()
    actions = asyncio.run(evolver.evolve(consumed, skill_manager.skills))
    assert len(actions) == 1
    assert actions[0].action == "create"

    # Step 4: Apply to skill manager
    skill_manager.add_skill(actions[0].skill)

    # Step 5: Verify skill exists
    new_skill = skill_manager.get_skill("check-logs-first")
    assert new_skill is not None
    assert new_skill["category"] == "coding"
    assert new_skill["content"].startswith("# check-logs-first")

    # Step 6: Verify skill is on disk
    skill_path = os.path.join(skill_manager._skills_dir, "check-logs-first", "SKILL.md")
    assert os.path.exists(skill_path)

    # Step 7: Verify skill is retrievable
    results = skill_manager.retrieve("debugging task", top_k=10)
    names = [s["name"] for s in results]
    assert "check-logs-first" in names


# --- Scenario 2: Correction accumulation flow ---

def test_correction_accumulation_triggers_evolution(detector, aggregator, evolver, skill_manager):
    """Three corrections accumulate before evolution triggers."""
    corrections = [
        "No, that's wrong. Show only the error line.",
        "No, don't include the full stack trace.",
        "No, that's wrong. Just the relevant line please.",
    ]

    for i, correction in enumerate(corrections):
        turn = {
            "session_id": f"sess-{i}",
            "turn_num": 1,
            "user_message": correction,
            "assistant_response": "Here's the full trace...",
            "active_skills": [],
        }
        signals = detector.detect(turn)
        assert len(signals) >= 1
        aggregator.add(signals)

    # Should trigger only after 3rd correction
    assert aggregator.should_evolve() is True

    # Evolve and apply
    consumed = aggregator.consume()
    assert len(consumed) == 3

    # Use a different FakeLLM for this scenario
    evolver_for_corrections = SkillEvolver(
        llm_client=FakeLLM(
            skill_name="concise-error-output",
            description="Show only the relevant error line",
        )
    )
    actions = asyncio.run(evolver_for_corrections.evolve(consumed, skill_manager.skills))
    assert len(actions) == 1
    skill_manager.add_skill(actions[0].skill)

    assert skill_manager.get_skill("concise-error-output") is not None


# --- Scenario 3: Feedback + confidence flow ---

def test_feedback_confidence_lifecycle(feedback_collector, aggregator, skill_manager):
    """Thumbs down → confidence drops → thumbs up → rises → deprecate → excluded."""
    initial = skill_manager.get_skill("existing-skill")
    assert initial["confidence"] == pytest.approx(0.6)

    # Thumbs down
    sig = feedback_collector.rate(["existing-skill"], positive=False, session_id="s1")
    aggregator.add([sig])
    skill_manager.adjust_confidence("existing-skill", -0.15)
    skill = skill_manager.get_skill("existing-skill")
    assert skill["confidence"] == pytest.approx(0.45, abs=0.01)

    # Thumbs up
    sig = feedback_collector.rate(["existing-skill"], positive=True, session_id="s2")
    aggregator.add([sig])
    skill_manager.adjust_confidence("existing-skill", 0.1)
    skill = skill_manager.get_skill("existing-skill")
    assert skill["confidence"] == pytest.approx(0.55, abs=0.01)

    # Still retrievable
    results = skill_manager.retrieve("coding task", top_k=10)
    assert any(s["name"] == "existing-skill" for s in results)

    # Deprecate via review
    signals = feedback_collector.submit_review([
        {"skill_name": "existing-skill", "action": "reject", "reason": "not useful"}
    ])
    assert signals[0].signal_type == "review_rejected"
    skill_manager.deprecate_skill("existing-skill")

    # Now excluded from retrieval
    results = skill_manager.retrieve("coding task", top_k=10)
    assert not any(s["name"] == "existing-skill" for s in results)
