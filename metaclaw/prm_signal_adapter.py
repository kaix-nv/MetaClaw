"""Wraps PRMScorer output into SkillSignal format for backward compatibility."""

from __future__ import annotations

from datetime import datetime

from .data_formatter import ConversationSample
from .skill_signal import SkillSignal


class PRMSignalAdapter:
    """Adapts PRMScorer results to SkillSignal format.

    Only adapts samples with reward != 0. Samples with reward=0
    (ambiguous) provide no actionable signal for skill evolution.
    """

    _CONFIDENCE_MAP = {-1.0: 1.0, 1.0: 1.0}

    def should_adapt(self, sample: ConversationSample) -> bool:
        return sample.reward != 0.0

    def adapt(self, sample: ConversationSample, active_skills: list[str]) -> SkillSignal:
        signal_type = "prm_failure" if sample.reward < 0 else "prm_success"
        return SkillSignal(
            source="prm",
            signal_type=signal_type,
            content={
                "prompt_text": sample.prompt_text,
                "response_text": sample.response_text,
                "reward": sample.reward,
                "active_skills": active_skills,
            },
            confidence=self._CONFIDENCE_MAP.get(sample.reward, 0.5),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            session_id=sample.session_id,
        )
