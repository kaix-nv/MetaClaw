"""Core data types for the conversation-driven skill evolution system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillSignal:
    """A normalized event from any signal source (conversation, feedback, PRM).

    All signal sources emit these; the evolver consumes only these.
    """

    source: str          # "conversation", "feedback", "prm"
    signal_type: str     # e.g. "explicit_save", "implicit_correction", "thumbs_up", "prm_failure"
    content: dict        # Source-specific payload
    confidence: float    # 0.0–1.0
    timestamp: str       # ISO 8601
    session_id: str      # Links back to conversation session


@dataclass
class SkillAction:
    """An action proposed by the SkillEvolver after analyzing signals."""

    action: str              # "create", "update", "deprecate", "link"
    skill: dict              # Skill dict: {name, description, content, category}
    parent_skill: str = ""   # For "link" action: parent skill name
    reasoning: str = ""      # LLM's reasoning
    source_signals: list[str] = field(default_factory=list)
