"""Merge evolved skills into a single markdown file for injection."""

from __future__ import annotations

import os
from pathlib import Path


class SkillWriter:
    """Produces a single merged skill file from base + evolved skills."""

    @staticmethod
    def merge(base_skill_path: str, evolved_skills: list[dict], output_path: str) -> None:
        """Concatenate base cuTile skill + evolved skills into one file."""
        parts = [Path(base_skill_path).read_text(encoding="utf-8")]

        if evolved_skills:
            parts.append("\n\n---\n\n# Evolved Skills (learned from TileGym)\n")
            for skill in evolved_skills:
                name = skill.get("name", "unnamed")
                desc = skill.get("description", "")
                content = skill.get("content", "")
                parts.append(f"\n## {name}\n_{desc}_\n\n{content}\n")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(parts), encoding="utf-8")
