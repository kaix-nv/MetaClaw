"""Detects skill-worthy signals from conversation turns."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

from .skill_signal import SkillSignal

logger = logging.getLogger(__name__)

_EXPLICIT_PATTERNS = [
    (re.compile(r"\bremember\s+this\b", re.I), 0.95),
    (re.compile(r"\bfrom\s+now\s+on\b", re.I), 0.95),
    (re.compile(r"\balways\s+(?:do|use|check|make sure)\b", re.I), 0.90),
    (re.compile(r"\bnever\s+(?:do|use|forget)\b", re.I), 0.90),
    (re.compile(r"\bsave\s+(?:this\s+)?as\s+a?\s*skill\b", re.I), 0.95),
    (re.compile(r"\bcreate\s+a\s+skill\b", re.I), 0.95),
    (re.compile(r"\badd\s+(?:this\s+)?to\s+your\s+skills\b", re.I), 0.95),
    (re.compile(r"\bkeep\s+doing\s+that\b", re.I), 0.85),
]

_CORRECTION_PATTERNS = [
    (re.compile(r"^no[,.\s]", re.I), 0.7),
    (re.compile(r"\bthat'?s\s+wrong\b", re.I), 0.8),
    (re.compile(r"\binstead\s+(?:of\s+that|,?\s*(?:you\s+)?should)\b", re.I), 0.7),
    (re.compile(r"\bactually[,\s]+(?:you\s+)?should\b", re.I), 0.7),
    (re.compile(r"\bdon'?t\s+do\s+(?:that|it\s+that\s+way)\b", re.I), 0.75),
    (re.compile(r"\bnot\s+what\s+I\s+(?:meant|asked|wanted)\b", re.I), 0.7),
]


class ConversationSignalDetector:
    def __init__(self, llm_client=None, use_llm_detection: bool = True):
        self._llm_client = llm_client
        self._use_llm = use_llm_detection
        self._correction_history: dict[str, list[SkillSignal]] = {}

    def detect(self, turn_data: dict) -> list[SkillSignal]:
        signals: list[SkillSignal] = []
        user_msg = turn_data.get("user_message", "")
        if not user_msg or len(user_msg.strip()) < 5:
            return signals
        signals.extend(self._detect_explicit(turn_data))
        if not signals:
            signals.extend(self._detect_implicit(turn_data))
        return signals

    def _detect_explicit(self, turn_data: dict) -> list[SkillSignal]:
        user_msg = turn_data["user_message"]
        now = datetime.now().isoformat(timespec="seconds")
        signals = []
        for pattern, confidence in _EXPLICIT_PATTERNS:
            if pattern.search(user_msg):
                signals.append(SkillSignal(
                    source="conversation",
                    signal_type="explicit_save",
                    content={
                        "user_message": user_msg,
                        "assistant_response": turn_data.get("assistant_response", ""),
                        "active_skills": turn_data.get("active_skills", []),
                    },
                    confidence=confidence,
                    timestamp=now,
                    session_id=turn_data.get("session_id", ""),
                ))
                break
        return signals

    def _detect_implicit(self, turn_data: dict) -> list[SkillSignal]:
        user_msg = turn_data["user_message"]
        now = datetime.now().isoformat(timespec="seconds")
        signals = []
        for pattern, confidence in _CORRECTION_PATTERNS:
            if pattern.search(user_msg):
                sig = SkillSignal(
                    source="conversation",
                    signal_type="implicit_correction",
                    content={
                        "user_message": user_msg,
                        "wrong_response": turn_data.get("assistant_response", ""),
                        "correction": user_msg,
                        "active_skills": turn_data.get("active_skills", []),
                    },
                    confidence=confidence,
                    timestamp=now,
                    session_id=turn_data.get("session_id", ""),
                )
                signals.append(sig)
                break
        return signals
