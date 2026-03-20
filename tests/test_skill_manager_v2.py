import os
import pytest
from metaclaw.skill_manager import SkillManager, _parse_skill_md


@pytest.fixture
def skill_dir(tmp_path):
    s1 = tmp_path / "test-alpha" / "SKILL.md"
    s1.parent.mkdir()
    s1.write_text(
        "---\nname: test-alpha\ndescription: Alpha skill\ncategory: coding\n---\n\n# Alpha\nDo alpha things.\n"
    )
    s2 = tmp_path / "test-beta" / "SKILL.md"
    s2.parent.mkdir()
    s2.write_text(
        "---\nname: test-beta\ndescription: Beta skill\n---\n\n# Beta\nDo beta things.\n"
    )
    return str(tmp_path)


@pytest.fixture
def mgr(skill_dir):
    return SkillManager(skill_dir, retrieval_mode="template")


def test_get_skill_found(mgr):
    skill = mgr.get_skill("test-alpha")
    assert skill is not None
    assert skill["name"] == "test-alpha"
    assert skill["category"] == "coding"


def test_get_skill_not_found(mgr):
    assert mgr.get_skill("nonexistent") is None


def test_iter_all_skills(mgr):
    names = [s["name"] for s in mgr._iter_all_skills()]
    assert "test-alpha" in names
    assert "test-beta" in names


def test_update_skill_content(mgr, skill_dir):
    assert mgr.update_skill("test-alpha", {"content": "# Updated\nNew content."})
    skill = mgr.get_skill("test-alpha")
    assert "New content" in skill["content"]
    ondisk = _parse_skill_md(os.path.join(skill_dir, "test-alpha", "SKILL.md"))
    assert "New content" in ondisk["content"]


def test_update_skill_not_found(mgr):
    assert mgr.update_skill("nonexistent", {"content": "x"}) is False


def test_deprecate_skill(mgr):
    assert mgr.deprecate_skill("test-alpha")
    skill = mgr.get_skill("test-alpha")
    assert skill["deprecated"] is True


def test_deprecate_skill_not_found(mgr):
    assert mgr.deprecate_skill("nonexistent") is False


def test_adjust_confidence_up(mgr):
    mgr.adjust_confidence("test-alpha", 0.1)
    skill = mgr.get_skill("test-alpha")
    assert skill["confidence"] == pytest.approx(0.7, abs=0.01)


def test_adjust_confidence_clamped(mgr):
    mgr.adjust_confidence("test-alpha", 2.0)
    skill = mgr.get_skill("test-alpha")
    assert skill["confidence"] == 1.0


def test_adjust_confidence_down(mgr):
    mgr.adjust_confidence("test-alpha", -0.8)
    skill = mgr.get_skill("test-alpha")
    assert skill["confidence"] == 0.0


def test_link_skills(mgr):
    assert mgr.link_skills("test-alpha", "test-beta")
    parent = mgr.get_skill("test-alpha")
    child = mgr.get_skill("test-beta")
    assert "test-beta" in parent["children"]
    assert child["parent"] == "test-alpha"


def test_link_skills_bad_parent(mgr):
    assert mgr.link_skills("nonexistent", "test-beta") is False


def test_parse_extended_frontmatter(tmp_path):
    s = tmp_path / "ext-skill" / "SKILL.md"
    s.parent.mkdir()
    s.write_text(
        "---\nname: ext-skill\ndescription: Extended\ncategory: coding\n"
        "parent: some-parent\nchildren: child-a, child-b\nconfidence: 0.9\n"
        "provenance: conversation\ncreated_from_session: sess-1\ndeprecated: false\n"
        "---\n\n# Extended\n"
    )
    skill = _parse_skill_md(str(s))
    assert skill["parent"] == "some-parent"
    assert skill["children"] == ["child-a", "child-b"]
    assert skill["confidence"] == pytest.approx(0.9)
    assert skill["provenance"] == "conversation"
    assert skill["deprecated"] is False


def test_retrieve_excludes_deprecated(mgr):
    mgr.deprecate_skill("test-alpha")
    results = mgr.retrieve("coding task", top_k=10)
    names = [s["name"] for s in results]
    assert "test-alpha" not in names


def test_retrieve_excludes_low_confidence(mgr):
    mgr.adjust_confidence("test-alpha", -0.5)
    results = mgr.retrieve("coding task", top_k=10)
    names = [s["name"] for s in results]
    assert "test-alpha" not in names
