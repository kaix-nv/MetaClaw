# tests/test_skill_writer.py
import os
from eval.cutile.skill_writer import SkillWriter


def test_merge_skills(tmp_path):
    base = tmp_path / "base.md"
    base.write_text("# cuTile Base Skill\nBase content here.\n")

    evolved = [
        {"name": "skill-a", "description": "Desc A", "content": "## A\nDo A."},
        {"name": "skill-b", "description": "Desc B", "content": "## B\nDo B."},
    ]

    output = tmp_path / "merged.md"
    SkillWriter.merge(str(base), evolved, str(output))

    text = output.read_text()
    assert "cuTile Base Skill" in text
    assert "Evolved Skills" in text
    assert "skill-a" in text
    assert "skill-b" in text
    assert "Do A." in text


def test_merge_no_evolved(tmp_path):
    base = tmp_path / "base.md"
    base.write_text("# Base\n")
    output = tmp_path / "merged.md"
    SkillWriter.merge(str(base), [], str(output))
    text = output.read_text()
    assert "# Base" in text
    assert "Evolved" not in text
