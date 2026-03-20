"""Collects structured user feedback and emits SkillSignals."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .skill_manager import SkillManager
from .skill_signal import SkillSignal

logger = logging.getLogger(__name__)


class FeedbackCollector:
    def __init__(self, skill_manager: SkillManager):
        self._skill_manager = skill_manager

    def rate(self, skill_names: list[str], positive: bool, session_id: str, response_id: str = "") -> SkillSignal:
        return SkillSignal(
            source="feedback",
            signal_type="thumbs_up" if positive else "thumbs_down",
            content={"skill_names": skill_names, "response_id": response_id},
            confidence=0.8,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            session_id=session_id,
        )

    def correct(self, skill_name: str, correction_text: str, session_id: str) -> SkillSignal:
        original = self._skill_manager.get_skill(skill_name)
        return SkillSignal(
            source="feedback",
            signal_type="user_correction",
            content={
                "skill_name": skill_name,
                "original": original.get("content", "") if original else "",
                "corrected": correction_text,
            },
            confidence=0.9,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            session_id=session_id,
        )

    def get_review_candidates(self, limit: int = 10) -> list[dict]:
        candidates = []
        for skill in self._skill_manager._iter_all_skills():
            if skill.get("deprecated", False):
                continue
            candidates.append(skill)
        candidates.sort(key=lambda s: s.get("confidence", 0.6))
        return candidates[:limit]

    def submit_review(self, reviews: list[dict]) -> list[SkillSignal]:
        signals = []
        now = datetime.now().isoformat(timespec="seconds")
        for review in reviews:
            name = review["skill_name"]
            action = review["action"]
            if action == "approve":
                signals.append(SkillSignal(
                    source="feedback", signal_type="review_approved",
                    content={"skill_name": name},
                    confidence=1.0, timestamp=now, session_id="",
                ))
            elif action == "reject":
                signals.append(SkillSignal(
                    source="feedback", signal_type="review_rejected",
                    content={"skill_name": name, "reason": review.get("reason", "")},
                    confidence=1.0, timestamp=now, session_id="",
                ))
            elif action == "edit":
                signals.append(SkillSignal(
                    source="feedback", signal_type="user_correction",
                    content={
                        "skill_name": name,
                        "original": "",
                        "corrected": review.get("edited_content", ""),
                    },
                    confidence=0.9, timestamp=now, session_id="",
                ))
        return signals
