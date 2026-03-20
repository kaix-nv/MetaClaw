from metaclaw.skill_signal import SkillSignal, SkillAction


def test_skill_signal_creation():
    sig = SkillSignal(
        source="conversation",
        signal_type="explicit_save",
        content={"user_message": "remember this"},
        confidence=0.95,
        timestamp="2026-03-20T10:00:00",
        session_id="sess-1",
    )
    assert sig.source == "conversation"
    assert sig.signal_type == "explicit_save"
    assert sig.confidence == 0.95


def test_skill_signal_defaults():
    sig = SkillSignal(
        source="prm",
        signal_type="prm_failure",
        content={},
        confidence=1.0,
        timestamp="2026-03-20T10:00:00",
        session_id="sess-1",
    )
    assert sig.session_id == "sess-1"


def test_skill_action_create():
    action = SkillAction(
        action="create",
        skill={"name": "test-skill", "description": "desc", "content": "body", "category": "general"},
        reasoning="test reason",
    )
    assert action.action == "create"
    assert action.skill["name"] == "test-skill"
    assert action.parent_skill == ""
    assert action.source_signals == []


def test_skill_action_link():
    action = SkillAction(
        action="link",
        skill={"name": "child-skill"},
        parent_skill="parent-skill",
    )
    assert action.parent_skill == "parent-skill"
