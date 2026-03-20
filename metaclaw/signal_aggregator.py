"""Buffers SkillSignals from all sources and triggers evolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .skill_signal import SkillSignal

logger = logging.getLogger(__name__)


@dataclass
class SkillEvolutionConfig:
    """Configuration for the signal-driven skill evolution system."""

    sources: list[str] = field(default_factory=lambda: ["conversation", "feedback"])
    correction_threshold: int = 3
    pattern_confidence_threshold: float = 0.8
    prm_failure_threshold: float = 0.4
    flush_interval_minutes: int = 30
    use_llm_detection: bool = True
    detection_model: str = ""
    enable_hierarchy: bool = True
    max_depth: int = 3
    initial_confidence: float = 0.6
    deprecation_threshold: float = 0.3


class SignalAggregator:
    """Aggregates SkillSignals and decides when to trigger evolution.

    Thread/async safety: consume() uses atomic swap to prevent signal loss.
    Signal source gating: only signals whose source is in config.sources are accepted.
    """

    def __init__(self, config: Optional[SkillEvolutionConfig] = None):
        self._config = config or SkillEvolutionConfig()
        self._buffer: list[SkillSignal] = []
        self._allowed_sources: set[str] = set(self._config.sources)

    def add(self, signals: list[SkillSignal]) -> None:
        """Add signals, filtering out sources not in config.sources."""
        self._buffer.extend(
            s for s in signals if s.source in self._allowed_sources
        )

    def should_evolve(self) -> bool:
        """Check if accumulated signals warrant skill evolution."""
        if not self._buffer:
            return False

        # 1. Any explicit_save -> immediate
        if any(s.signal_type == "explicit_save" for s in self._buffer):
            return True

        # 2. Correction count >= threshold
        corrections = sum(
            1 for s in self._buffer if s.signal_type == "implicit_correction"
        )
        if corrections >= self._config.correction_threshold:
            return True

        # 3. High-confidence pattern detected
        if any(
            s.signal_type == "pattern_detected"
            and s.confidence >= self._config.pattern_confidence_threshold
            for s in self._buffer
        ):
            return True

        # 4. PRM failure rate (count-based for buffered signals)
        prm_signals = [s for s in self._buffer if s.source == "prm"]
        if prm_signals:
            failures = sum(1 for s in prm_signals if s.signal_type == "prm_failure")
            if len(prm_signals) > 0 and failures / len(prm_signals) > (1 - self._config.prm_failure_threshold):
                return True

        return False

    def consume(self) -> list[SkillSignal]:
        """Return and clear buffered signals. Atomic swap prevents signal loss."""
        signals, self._buffer = self._buffer, []
        return signals
