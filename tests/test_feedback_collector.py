import pytest
from metaclaw.feedback_collector import FeedbackCollector
from metaclaw.skill_manager import SkillManager


@pytest.fixture
def skill_dir(tmp_path):
    s1 = tmp_path / "test-skill" / "SKILL.md"
    s1.parent.mkdir()
    s1.write_text(
        "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test\nOriginal content.\n"
    )
    return str(tmp_path)


@pytest.fixture
def mgr(skill_dir):
    return SkillManager(skill_dir, retrieval_mode="template")


@pytest.fixture
def collector(mgr):
    return FeedbackCollector(mgr)


def test_rate_positive(collector):
    sig = collector.rate(["test-skill"], positive=True, session_id="s1")
    assert sig.signal_type == "thumbs_up"
    assert sig.source == "feedback"


def test_rate_negative(collector):
    sig = collector.rate(["test-skill"], positive=False, session_id="s1")
    assert sig.signal_type == "thumbs_down"


def test_correct_with_original(collector):
    sig = collector.correct("test-skill", "New corrected content", session_id="s1")
    assert sig.signal_type == "user_correction"
    assert sig.content["original"] == "# Test\nOriginal content."
    assert sig.content["corrected"] == "New corrected content"


def test_correct_missing_skill(collector):
    sig = collector.correct("nonexistent", "fix", session_id="s1")
    assert sig.content["original"] == ""


def test_get_review_candidates(collector, mgr):
    mgr.adjust_confidence("test-skill", -0.4)
    candidates = collector.get_review_candidates(limit=5)
    assert len(candidates) >= 1
    assert candidates[0]["name"] == "test-skill"


def test_submit_review_approve(collector):
    signals = collector.submit_review([{"skill_name": "test-skill", "action": "approve"}])
    assert len(signals) == 1
    assert signals[0].signal_type == "review_approved"


def test_submit_review_reject(collector):
    signals = collector.submit_review([{"skill_name": "test-skill", "action": "reject", "reason": "not useful"}])
    assert len(signals) == 1
    assert signals[0].signal_type == "review_rejected"
