# Conversation-Driven Skill Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add conversation-driven skill extraction and feedback-based iteration to MetaClaw, so agents improve through chat without dataset collection or model training.

**Architecture:** Pluggable signal sources (conversation detection, user feedback, PRM) feed a unified `SignalAggregator` that triggers `SkillEvolver` to create/update/deprecate/link skills via LLM analysis. Built as a MetaClaw extension on the existing `SkillManager` + `SkillEvolver`.

**Tech Stack:** Python 3.10+, FastAPI, pytest, pytest-asyncio, dataclasses.

**New dev dependency:** `pytest-asyncio` (needed for Task 7 async tests). Install: `pip install pytest-asyncio`. The existing test suite uses `asyncio.run()` wrappers but the refactored evolver tests use `@pytest.mark.asyncio` for cleaner async test patterns.

**Spec:** `docs/specs/2026-03-20-conversation-driven-skill-evolution-design.md`

---

## File Structure

```
metaclaw/
├── skill_signal.py                  [NEW]  SkillSignal + SkillAction dataclasses
├── signal_aggregator.py             [NEW]  Buffers signals, triggers evolution
├── conversation_signal_detector.py  [NEW]  Explicit + implicit signal detection
├── feedback_collector.py            [NEW]  REST endpoints for structured feedback
├── prm_signal_adapter.py            [NEW]  Wraps PRMScorer output → SkillSignal
├── skill_evolver.py                 [MOD]  Refactored: SkillSignal[] input, SkillAction[] output
├── skill_manager.py                 [MOD]  +get_skill, +update_skill, +deprecate, +hierarchy, +confidence
├── api_server.py                    [MOD]  +signal detector hook, +feedback endpoints
├── trainer.py                       [MOD]  +_apply_skill_actions, +signal aggregator wiring
├── config.py                        [MOD]  +SkillEvolutionConfig fields

tests/
├── test_skill_signal.py             [NEW]
├── test_signal_aggregator.py        [NEW]
├── test_conversation_signal_detector.py  [NEW]
├── test_feedback_collector.py       [NEW]
├── test_prm_signal_adapter.py       [NEW]
├── test_skill_evolver_v2.py         [NEW]  Tests for refactored evolver
├── test_skill_manager_v2.py         [NEW]  Tests for new SkillManager methods
```

---

### Task 1: SkillSignal and SkillAction dataclasses

**Files:**
- Create: `metaclaw/skill_signal.py`
- Test: `tests/test_skill_signal.py`

- [ ] **Step 1: Write tests for SkillSignal and SkillAction**

```python
# tests/test_skill_signal.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_skill_signal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'metaclaw.skill_signal'`

- [ ] **Step 3: Implement skill_signal.py**

```python
# metaclaw/skill_signal.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_skill_signal.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add metaclaw/skill_signal.py tests/test_skill_signal.py
git commit -m "feat: add SkillSignal and SkillAction dataclasses"
```

---

### Task 2: SkillManager extensions (get_skill, update, deprecate, confidence, hierarchy)

**Files:**
- Modify: `metaclaw/skill_manager.py`
- Test: `tests/test_skill_manager_v2.py`

- [ ] **Step 1: Write tests for new SkillManager methods**

```python
# tests/test_skill_manager_v2.py
import os
import tempfile

import pytest

from metaclaw.skill_manager import SkillManager, _parse_skill_md


@pytest.fixture
def skill_dir(tmp_path):
    """Create a temp skills dir with two skills."""
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


# --- get_skill ---

def test_get_skill_found(mgr):
    skill = mgr.get_skill("test-alpha")
    assert skill is not None
    assert skill["name"] == "test-alpha"
    assert skill["category"] == "coding"


def test_get_skill_not_found(mgr):
    assert mgr.get_skill("nonexistent") is None


# --- _iter_all_skills ---

def test_iter_all_skills(mgr):
    names = [s["name"] for s in mgr._iter_all_skills()]
    assert "test-alpha" in names
    assert "test-beta" in names


# --- update_skill ---

def test_update_skill_content(mgr, skill_dir):
    assert mgr.update_skill("test-alpha", {"content": "# Updated\nNew content."})
    skill = mgr.get_skill("test-alpha")
    assert "New content" in skill["content"]
    # Verify persisted to disk
    ondisk = _parse_skill_md(os.path.join(skill_dir, "test-alpha", "SKILL.md"))
    assert "New content" in ondisk["content"]


def test_update_skill_not_found(mgr):
    assert mgr.update_skill("nonexistent", {"content": "x"}) is False


# --- deprecate_skill ---

def test_deprecate_skill(mgr):
    assert mgr.deprecate_skill("test-alpha")
    skill = mgr.get_skill("test-alpha")
    assert skill["deprecated"] is True


def test_deprecate_skill_not_found(mgr):
    assert mgr.deprecate_skill("nonexistent") is False


# --- adjust_confidence ---

def test_adjust_confidence_up(mgr):
    mgr.adjust_confidence("test-alpha", 0.1)
    skill = mgr.get_skill("test-alpha")
    assert skill["confidence"] == pytest.approx(0.7, abs=0.01)  # 0.6 default + 0.1


def test_adjust_confidence_clamped(mgr):
    mgr.adjust_confidence("test-alpha", 2.0)
    skill = mgr.get_skill("test-alpha")
    assert skill["confidence"] == 1.0


def test_adjust_confidence_down(mgr):
    mgr.adjust_confidence("test-alpha", -0.8)
    skill = mgr.get_skill("test-alpha")
    assert skill["confidence"] == 0.0


# --- hierarchy ---

def test_link_skills(mgr):
    assert mgr.link_skills("test-alpha", "test-beta")
    parent = mgr.get_skill("test-alpha")
    child = mgr.get_skill("test-beta")
    assert "test-beta" in parent["children"]
    assert child["parent"] == "test-alpha"


def test_link_skills_bad_parent(mgr):
    assert mgr.link_skills("nonexistent", "test-beta") is False


# --- parse extended frontmatter ---

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


# --- retrieve filters deprecated and low-confidence ---

def test_retrieve_excludes_deprecated(mgr):
    mgr.deprecate_skill("test-alpha")
    results = mgr.retrieve("coding task", top_k=10)
    names = [s["name"] for s in results]
    assert "test-alpha" not in names


def test_retrieve_excludes_low_confidence(mgr):
    mgr.adjust_confidence("test-alpha", -0.5)  # 0.6 - 0.5 = 0.1 < 0.3 threshold
    results = mgr.retrieve("coding task", top_k=10)
    names = [s["name"] for s in results]
    assert "test-alpha" not in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_skill_manager_v2.py -v`
Expected: FAIL — `AttributeError: 'SkillManager' object has no attribute 'get_skill'`

- [ ] **Step 3: Extend `_parse_skill_md()` to parse new frontmatter fields**

In `metaclaw/skill_manager.py`, modify `_parse_skill_md()` (lines 94–138). After line 127 add new field extraction. Change the return dict (lines 133–138) to include the new fields. Add a `_parse_list()` helper before `_parse_skill_md()`.

```python
def _parse_list(val: str) -> list[str]:
    """Parse 'a, b, c' or '[a, b, c]' into a list of stripped strings."""
    if not val:
        return []
    val = val.strip().strip("[]")
    return [v.strip() for v in val.split(",") if v.strip()]
```

Extend the return dict:
```python
    return {
        "name": name,
        "description": description,
        "category": category,
        "content": body,
        "parent": fm.get("parent", None) or None,
        "children": _parse_list(fm.get("children", "")),
        "confidence": float(fm.get("confidence", 0.6)),
        "provenance": fm.get("provenance", "manual"),
        "created_from_session": fm.get("created_from_session", ""),
        "deprecated": fm.get("deprecated", "false").lower() == "true",
    }
```

- [ ] **Step 4: Extend `_write_skill_md()` to write new frontmatter fields**

In `metaclaw/skill_manager.py`, modify `_write_skill_md()` (lines 424–448). Replace the `fm_lines` construction block (lines 438–441) with the following expanded version. The new fields must be inserted **before** `fm = "\n".join(fm_lines)` (line 441):

```python
        fm_lines = [f"name: {name}", f"description: {description}"]
        if category and category != "general":
            fm_lines.append(f"category: {category}")
        # New fields — only write non-default values
        parent = skill.get("parent")
        if parent:
            fm_lines.append(f"parent: {parent}")
        children = skill.get("children", [])
        if children:
            fm_lines.append(f"children: {', '.join(children)}")
        confidence = skill.get("confidence")
        if confidence is not None and confidence != 0.6:
            fm_lines.append(f"confidence: {confidence:.2f}")
        provenance = skill.get("provenance", "manual")
        if provenance != "manual":
            fm_lines.append(f"provenance: {provenance}")
        session = skill.get("created_from_session", "")
        if session:
            fm_lines.append(f"created_from_session: {session}")
        if skill.get("deprecated"):
            fm_lines.append("deprecated: true")
```

- [ ] **Step 5: Add new methods to SkillManager class**

Add after `get_skill_count()` (after line 487):

```python
    def get_skill(self, name: str) -> Optional[dict]:
        """Return the full skill dict by name, or None if not found."""
        for skill in self._iter_all_skills():
            if skill.get("name") == name:
                return skill
        return None

    def _iter_all_skills(self):
        """Iterate over all skills across all categories."""
        yield from self.skills.get("general_skills", [])
        for cat_skills in self.skills.get("task_specific_skills", {}).values():
            yield from cat_skills
        yield from self.skills.get("common_mistakes", [])

    def update_skill(self, name: str, updates: dict) -> bool:
        """Update an existing skill's fields and persist to disk."""
        skill = self.get_skill(name)
        if skill is None:
            logger.warning("[SkillManager] update_skill: not found: %s", name)
            return False
        for key, val in updates.items():
            if key != "name":  # name is immutable
                skill[key] = val
        self._skill_embeddings_cache = None
        self._write_skill_md(skill)
        logger.info("[SkillManager] updated skill: %s", name)
        return True

    def deprecate_skill(self, name: str, reason: str = "") -> bool:
        """Mark a skill as deprecated. Excluded from retrieval but not deleted."""
        skill = self.get_skill(name)
        if skill is None:
            logger.warning("[SkillManager] deprecate_skill: not found: %s", name)
            return False
        skill["deprecated"] = True
        self._skill_embeddings_cache = None
        self._write_skill_md(skill)
        logger.info("[SkillManager] deprecated skill: %s (reason: %s)", name, reason)
        return True

    def adjust_confidence(self, name: str, delta: float) -> None:
        """Adjust a skill's confidence score, clamped to [0.0, 1.0], and persist."""
        skill = self.get_skill(name)
        if skill is None:
            return
        current = skill.get("confidence", 0.6)
        skill["confidence"] = max(0.0, min(1.0, current + delta))
        self._write_skill_md(skill)

    def link_skills(self, parent_name: str, child_name: str) -> bool:
        """Create parent-child relationship. Updates both skills and persists."""
        parent = self.get_skill(parent_name)
        child = self.get_skill(child_name)
        if parent is None:
            logger.warning("[SkillManager] link_skills: parent not found: %s", parent_name)
            return False
        if child is None:
            logger.warning("[SkillManager] link_skills: child not found: %s", child_name)
            return False
        children = parent.get("children", [])
        if child_name not in children:
            children.append(child_name)
        parent["children"] = children
        child["parent"] = parent_name
        self._write_skill_md(parent)
        self._write_skill_md(child)
        return True
```

- [ ] **Step 6: Modify `retrieve()` to filter deprecated and low-confidence skills**

In `metaclaw/skill_manager.py`, modify `retrieve()` (lines 336–353). Add a filter step before returning. Replace the final return line:

```python
    def retrieve(self, task_description: str, top_k: int = 6) -> list[dict]:
        """Retrieve relevant skills for *task_description*."""
        common_mistakes = self.skills.get("common_mistakes", [])[:5]

        if self.retrieval_mode == "embedding":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            general, task_skills = self._embedding_retrieve(task_description, top_k, ts_top_k)
        else:
            task_type = self._detect_task_type(task_description)
            general = self.skills.get("general_skills", [])[:top_k]
            all_task = self.skills.get("task_specific_skills", {}).get(task_type, [])
            task_skills = (
                all_task[: self.task_specific_top_k]
                if self.task_specific_top_k is not None
                else all_task
            )

        combined = general + task_skills + common_mistakes
        # Filter out deprecated and low-confidence skills
        return [
            s for s in combined
            if not s.get("deprecated", False)
            and s.get("confidence", 0.6) >= 0.3
        ]
```

- [ ] **Step 7: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_skill_manager_v2.py -v`
Expected: All tests PASS

- [ ] **Step 8: Run existing tests to check for regressions**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/ -v --ignore=tests/test_v03_live_tinker.py`
Expected: No regressions in existing tests

- [ ] **Step 9: Commit**

```bash
git add metaclaw/skill_manager.py tests/test_skill_manager_v2.py
git commit -m "feat: extend SkillManager with get/update/deprecate/confidence/hierarchy"
```

---

### Task 3: SignalAggregator

**Files:**
- Create: `metaclaw/signal_aggregator.py`
- Test: `tests/test_signal_aggregator.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_signal_aggregator.py
from metaclaw.signal_aggregator import SignalAggregator, SkillEvolutionConfig
from metaclaw.skill_signal import SkillSignal


def _sig(source="conversation", signal_type="explicit_save", confidence=0.9):
    return SkillSignal(
        source=source,
        signal_type=signal_type,
        content={},
        confidence=confidence,
        timestamp="2026-03-20T10:00:00",
        session_id="s1",
    )


def test_add_and_consume():
    agg = SignalAggregator(SkillEvolutionConfig())
    agg.add([_sig(), _sig()])
    signals = agg.consume()
    assert len(signals) == 2
    assert agg.consume() == []  # buffer cleared


def test_source_filtering():
    """Only sources listed in config.sources are accepted."""
    cfg = SkillEvolutionConfig(sources=["conversation"])
    agg = SignalAggregator(cfg)
    agg.add([_sig(source="conversation"), _sig(source="prm")])
    signals = agg.consume()
    assert len(signals) == 1
    assert signals[0].source == "conversation"


def test_should_evolve_explicit_save():
    agg = SignalAggregator(SkillEvolutionConfig())
    agg.add([_sig(signal_type="explicit_save")])
    assert agg.should_evolve() is True


def test_should_evolve_correction_threshold():
    cfg = SkillEvolutionConfig(correction_threshold=3)
    agg = SignalAggregator(cfg)
    agg.add([_sig(signal_type="implicit_correction")])
    assert agg.should_evolve() is False
    agg.add([_sig(signal_type="implicit_correction")])
    assert agg.should_evolve() is False
    agg.add([_sig(signal_type="implicit_correction")])
    assert agg.should_evolve() is True


def test_should_evolve_pattern_high_confidence():
    agg = SignalAggregator(SkillEvolutionConfig(pattern_confidence_threshold=0.8))
    agg.add([_sig(signal_type="pattern_detected", confidence=0.9)])
    assert agg.should_evolve() is True


def test_should_evolve_empty():
    agg = SignalAggregator(SkillEvolutionConfig())
    assert agg.should_evolve() is False


def test_consume_atomic_swap():
    """consume() returns the buffer and replaces it atomically."""
    agg = SignalAggregator(SkillEvolutionConfig())
    agg.add([_sig()])
    old_buffer = agg._buffer
    consumed = agg.consume()
    assert consumed is old_buffer
    assert agg._buffer is not old_buffer
    assert agg._buffer == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_signal_aggregator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement signal_aggregator.py**

```python
# metaclaw/signal_aggregator.py
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

        # 1. Any explicit_save → immediate
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
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_signal_aggregator.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add metaclaw/signal_aggregator.py tests/test_signal_aggregator.py
git commit -m "feat: add SignalAggregator with source filtering and trigger conditions"
```

---

### Task 4: ConversationSignalDetector

**Files:**
- Create: `metaclaw/conversation_signal_detector.py`
- Test: `tests/test_conversation_signal_detector.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_conversation_signal_detector.py
from metaclaw.conversation_signal_detector import ConversationSignalDetector


def _turn(user_msg, assistant_resp="OK", session_id="s1", turn_num=1):
    return {
        "session_id": session_id,
        "turn_num": turn_num,
        "user_message": user_msg,
        "assistant_response": assistant_resp,
        "active_skills": [],
    }


# --- Explicit detection ---

def test_explicit_remember_this():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Please remember this: always use UTC timestamps"))
    assert len(signals) >= 1
    assert signals[0].signal_type == "explicit_save"
    assert signals[0].confidence >= 0.9


def test_explicit_from_now_on():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("From now on, always check logs first"))
    assert len(signals) >= 1
    assert signals[0].signal_type == "explicit_save"


def test_explicit_save_as_skill():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Save this as a skill: verify file paths"))
    assert len(signals) >= 1
    assert signals[0].signal_type == "explicit_save"


def test_explicit_never_do():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Never do that again, always ask before deleting"))
    assert len(signals) >= 1
    assert signals[0].signal_type == "explicit_save"


# --- Implicit correction detection ---

def test_implicit_correction_no_wrong():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("No, that's wrong. You should use async instead"))
    corrections = [s for s in signals if s.signal_type == "implicit_correction"]
    assert len(corrections) >= 1
    assert corrections[0].confidence >= 0.5


def test_implicit_correction_instead():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Instead of that, use a context manager"))
    corrections = [s for s in signals if s.signal_type == "implicit_correction"]
    assert len(corrections) >= 1


# --- No false positives ---

def test_no_signal_on_normal_message():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Can you help me write a function?"))
    assert len(signals) == 0


def test_no_signal_on_short_message():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("ok"))
    assert len(signals) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_conversation_signal_detector.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement conversation_signal_detector.py**

```python
# metaclaw/conversation_signal_detector.py
"""Detects skill-worthy signals from conversation turns."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

from .skill_signal import SkillSignal

logger = logging.getLogger(__name__)

# Explicit trigger patterns (case-insensitive)
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

# Implicit correction heuristic patterns
_CORRECTION_PATTERNS = [
    (re.compile(r"^no[,.\s]", re.I), 0.7),
    (re.compile(r"\bthat'?s\s+wrong\b", re.I), 0.8),
    (re.compile(r"\binstead\s+(?:of\s+that|,?\s*(?:you\s+)?should)\b", re.I), 0.7),
    (re.compile(r"\bactually[,\s]+(?:you\s+)?should\b", re.I), 0.7),
    (re.compile(r"\bdon'?t\s+do\s+(?:that|it\s+that\s+way)\b", re.I), 0.75),
    (re.compile(r"\bnot\s+what\s+I\s+(?:meant|asked|wanted)\b", re.I), 0.7),
]


class ConversationSignalDetector:
    """Detects skill-worthy signals from conversation turns.

    Explicit detection: keyword/pattern matching (fast, no LLM).
    Implicit detection: heuristic correction patterns.
    LLM confirmation pass: optional, reduces false positives (not implemented in v1).
    """

    def __init__(
        self,
        llm_client=None,
        use_llm_detection: bool = True,
    ):
        self._llm_client = llm_client
        self._use_llm = use_llm_detection
        self._correction_history: dict[str, list[SkillSignal]] = {}

    def detect(self, turn_data: dict) -> list[SkillSignal]:
        """Analyze a conversation turn for skill signals.

        turn_data keys:
            session_id, turn_num, user_message, assistant_response, active_skills[]
        """
        signals: list[SkillSignal] = []
        user_msg = turn_data.get("user_message", "")
        if not user_msg or len(user_msg.strip()) < 5:
            return signals

        signals.extend(self._detect_explicit(turn_data))
        # Only check implicit if no explicit found (avoid double-signaling)
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
                break  # One explicit signal per turn is enough
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
                break  # One correction signal per turn
        return signals
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_conversation_signal_detector.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add metaclaw/conversation_signal_detector.py tests/test_conversation_signal_detector.py
git commit -m "feat: add ConversationSignalDetector with explicit and implicit detection"
```

---

### Task 5: FeedbackCollector

**Files:**
- Create: `metaclaw/feedback_collector.py`
- Test: `tests/test_feedback_collector.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_feedback_collector.py
import os
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
    # Lower confidence so it appears in review
    mgr.adjust_confidence("test-skill", -0.4)
    candidates = collector.get_review_candidates(limit=5)
    assert len(candidates) >= 1
    assert candidates[0]["name"] == "test-skill"


def test_submit_review_approve(collector):
    signals = collector.submit_review([
        {"skill_name": "test-skill", "action": "approve"}
    ])
    assert len(signals) == 1
    assert signals[0].signal_type == "review_approved"


def test_submit_review_reject(collector):
    signals = collector.submit_review([
        {"skill_name": "test-skill", "action": "reject", "reason": "not useful"}
    ])
    assert len(signals) == 1
    assert signals[0].signal_type == "review_rejected"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_feedback_collector.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement feedback_collector.py**

```python
# metaclaw/feedback_collector.py
"""Collects structured user feedback and emits SkillSignals."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .skill_manager import SkillManager
from .skill_signal import SkillSignal

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects structured user feedback and converts to SkillSignals."""

    def __init__(self, skill_manager: SkillManager):
        self._skill_manager = skill_manager

    def rate(
        self,
        skill_names: list[str],
        positive: bool,
        session_id: str,
        response_id: str = "",
    ) -> SkillSignal:
        """Create a thumbs_up or thumbs_down signal."""
        return SkillSignal(
            source="feedback",
            signal_type="thumbs_up" if positive else "thumbs_down",
            content={
                "skill_names": skill_names,
                "response_id": response_id,
            },
            confidence=0.8,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            session_id=session_id,
        )

    def correct(
        self,
        skill_name: str,
        correction_text: str,
        session_id: str,
    ) -> SkillSignal:
        """Create a user_correction signal with the original skill content."""
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
        """Return skills with low confidence, sorted ascending."""
        candidates = []
        for skill in self._skill_manager._iter_all_skills():
            if skill.get("deprecated", False):
                continue
            candidates.append(skill)
        candidates.sort(key=lambda s: s.get("confidence", 0.6))
        return candidates[:limit]

    def submit_review(self, reviews: list[dict]) -> list[SkillSignal]:
        """Process batch review results.

        Each review: {"skill_name": str, "action": "approve"|"edit"|"reject",
                      "edited_content": str (for edit), "reason": str (for reject)}
        """
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
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_feedback_collector.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add metaclaw/feedback_collector.py tests/test_feedback_collector.py
git commit -m "feat: add FeedbackCollector with rate, correct, and review"
```

---

### Task 6: PRMSignalAdapter

**Files:**
- Create: `metaclaw/prm_signal_adapter.py`
- Test: `tests/test_prm_signal_adapter.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_prm_signal_adapter.py
from metaclaw.prm_signal_adapter import PRMSignalAdapter
from metaclaw.data_formatter import ConversationSample


def _sample(reward: float) -> ConversationSample:
    return ConversationSample(
        session_id="s1",
        turn_num=1,
        prompt_tokens=[1, 2, 3],
        response_tokens=[4, 5],
        response_logprobs=[-0.5, -0.3],
        loss_mask=[1, 1],
        reward=reward,
        prompt_text="hello",
        response_text="world",
    )


def test_should_adapt_failure():
    adapter = PRMSignalAdapter()
    assert adapter.should_adapt(_sample(-1.0)) is True


def test_should_adapt_success():
    adapter = PRMSignalAdapter()
    assert adapter.should_adapt(_sample(1.0)) is True


def test_should_not_adapt_ambiguous():
    adapter = PRMSignalAdapter()
    assert adapter.should_adapt(_sample(0.0)) is False


def test_adapt_failure():
    adapter = PRMSignalAdapter()
    sig = adapter.adapt(_sample(-1.0), ["skill-a"])
    assert sig.source == "prm"
    assert sig.signal_type == "prm_failure"
    assert sig.confidence == 1.0
    assert sig.content["active_skills"] == ["skill-a"]


def test_adapt_success():
    adapter = PRMSignalAdapter()
    sig = adapter.adapt(_sample(1.0), [])
    assert sig.signal_type == "prm_success"
    assert sig.confidence == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_prm_signal_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement prm_signal_adapter.py**

```python
# metaclaw/prm_signal_adapter.py
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
        """Return False for ambiguous samples (reward=0)."""
        return sample.reward != 0.0

    def adapt(
        self,
        sample: ConversationSample,
        active_skills: list[str],
    ) -> SkillSignal:
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
```

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_prm_signal_adapter.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add metaclaw/prm_signal_adapter.py tests/test_prm_signal_adapter.py
git commit -m "feat: add PRMSignalAdapter for backward-compatible signal conversion"
```

---

### Task 7: Refactor SkillEvolver to consume SkillSignal[]

**Files:**
- Modify: `metaclaw/skill_evolver.py`
- Test: `tests/test_skill_evolver_v2.py`

**Prerequisite:** Install `pytest-asyncio`:
```bash
pip install pytest-asyncio
```
Add `asyncio_mode = "auto"` to `pyproject.toml` under `[tool.pytest.ini_options]` if the section exists, or create it.

- [ ] **Step 1: Write tests for the refactored evolver**

```python
# tests/test_skill_evolver_v2.py
import json
import pytest
from metaclaw.skill_evolver import SkillEvolver
from metaclaw.skill_signal import SkillSignal, SkillAction


class FakeLLMClient:
    """Returns a canned JSON response for testing."""

    def __init__(self, response: str):
        self._response = response

    def chat_complete(self, prompt: str) -> str:
        return self._response


def _sig(signal_type="explicit_save", content=None):
    return SkillSignal(
        source="conversation",
        signal_type=signal_type,
        content=content or {"user_message": "test"},
        confidence=0.9,
        timestamp="2026-03-20T10:00:00",
        session_id="s1",
    )


def _skills():
    return {
        "general_skills": [{"name": "existing-skill", "description": "exists"}],
        "task_specific_skills": {},
        "common_mistakes": [],
    }


@pytest.mark.asyncio
async def test_evolve_creates_skill():
    llm_response = json.dumps([{
        "action": "create",
        "skill": {
            "name": "new-skill",
            "description": "A new skill",
            "content": "# New\nDo this.",
            "category": "general",
        },
        "reasoning": "User explicitly asked",
    }])
    evolver = SkillEvolver(llm_client=FakeLLMClient(llm_response))
    actions = await evolver.evolve([_sig()], _skills())
    assert len(actions) == 1
    assert isinstance(actions[0], SkillAction)
    assert actions[0].action == "create"
    assert actions[0].skill["name"] == "new-skill"


@pytest.mark.asyncio
async def test_evolve_empty_signals():
    evolver = SkillEvolver(llm_client=FakeLLMClient("[]"))
    actions = await evolver.evolve([], _skills())
    assert actions == []


@pytest.mark.asyncio
async def test_evolve_update_action():
    llm_response = json.dumps([{
        "action": "update",
        "skill": {"name": "existing-skill", "description": "updated", "content": "new", "category": "general"},
        "reasoning": "corrected",
    }])
    evolver = SkillEvolver(llm_client=FakeLLMClient(llm_response))
    actions = await evolver.evolve([_sig("implicit_correction")], _skills())
    assert actions[0].action == "update"


@pytest.mark.asyncio
async def test_evolve_link_action():
    llm_response = json.dumps([{
        "action": "link",
        "skill": {"name": "child-skill"},
        "parent_skill": "existing-skill",
        "reasoning": "fits as sub-skill",
    }])
    evolver = SkillEvolver(llm_client=FakeLLMClient(llm_response))
    actions = await evolver.evolve([_sig()], _skills())
    assert actions[0].action == "link"
    assert actions[0].parent_skill == "existing-skill"


@pytest.mark.asyncio
async def test_evolve_llm_failure_returns_empty():
    class FailClient:
        def chat_complete(self, prompt):
            raise RuntimeError("LLM down")

    evolver = SkillEvolver(llm_client=FailClient())
    actions = await evolver.evolve([_sig()], _skills())
    assert actions == []


@pytest.mark.asyncio
async def test_evolve_passes_existing_skills_to_prompt():
    """Verify the prompt includes existing skill names for deduplication."""
    captured = {}

    class CapturingClient:
        def chat_complete(self, prompt):
            captured["prompt"] = prompt
            return "[]"

    evolver = SkillEvolver(llm_client=CapturingClient())
    await evolver.evolve([_sig()], _skills())
    assert "existing-skill" in captured["prompt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_skill_evolver_v2.py -v`
Expected: FAIL — `evolve()` signature mismatch or missing `SkillAction` return

- [ ] **Step 3: Refactor `skill_evolver.py`**

Key changes to `metaclaw/skill_evolver.py`:
1. Add import for `SkillSignal`, `SkillAction` at top
2. Change `evolve()` signature: `failed_samples: list` → `signals: list[SkillSignal]`, return `list[SkillAction]`
3. Replace `_build_analysis_prompt()` with `_build_signal_analysis_prompt()`
4. Replace `_parse_skills_response()` with `_parse_actions_response()`
5. Keep `should_evolve()` as deprecated (backward compat) but it's no longer the primary trigger
6. Keep `_call_llm()`, `_finalise_names()`, `_next_dyn_index()`, `_append_history()` unchanged

The `_build_signal_analysis_prompt()` method groups signals by type and formats them, then appends the existing skill inventory for deduplication (same logic as old `_build_analysis_prompt` lines 246–259).

The `_parse_actions_response()` method parses JSON array of action objects, each with `action`, `skill`, `parent_skill`, `reasoning` keys, and returns `list[SkillAction]`.

- [ ] **Step 4: Run tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/test_skill_evolver_v2.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run all tests for regression**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/ -v --ignore=tests/test_v03_live_tinker.py`
Expected: No regressions

- [ ] **Step 6: Commit**

```bash
git add metaclaw/skill_evolver.py tests/test_skill_evolver_v2.py
git commit -m "refactor: SkillEvolver consumes SkillSignal[], returns SkillAction[]"
```

---

### Task 8: Config extensions

**Files:**
- Modify: `metaclaw/config.py`

- [ ] **Step 1: Add SkillEvolutionConfig fields to MetaClawConfig**

In `metaclaw/config.py`, add new fields after the existing skills section (after line 64). These are flat fields on `MetaClawConfig` (no nested dataclass) to stay consistent with the existing config style:

```python
    # ------------------------------------------------------------------ #
    # Conversation-driven skill evolution                                  #
    # ------------------------------------------------------------------ #
    skill_evolution_sources: str = "conversation,feedback"  # comma-separated: "conversation", "feedback", "prm"
    skill_evolution_correction_threshold: int = 3
    skill_evolution_pattern_confidence: float = 0.8
    skill_evolution_use_llm_detection: bool = True
    skill_evolution_initial_confidence: float = 0.6
    skill_evolution_deprecation_threshold: float = 0.3
```

**Note:** `SkillEvolutionConfig` is defined in `metaclaw/signal_aggregator.py` (Task 3) to keep it co-located with its primary consumer. The `skill_evolution_config()` method below provides conversion from flat `MetaClawConfig` fields.

- [ ] **Step 2: Add helper to convert flat config to SkillEvolutionConfig**

At the bottom of `config.py`, add:

```python
    def skill_evolution_config(self):
        """Convert flat config fields to a SkillEvolutionConfig instance."""
        from .signal_aggregator import SkillEvolutionConfig
        return SkillEvolutionConfig(
            sources=[s.strip() for s in self.skill_evolution_sources.split(",") if s.strip()],
            correction_threshold=self.skill_evolution_correction_threshold,
            pattern_confidence_threshold=self.skill_evolution_pattern_confidence,
            prm_failure_threshold=self.skill_update_threshold,
            use_llm_detection=self.skill_evolution_use_llm_detection,
            initial_confidence=self.skill_evolution_initial_confidence,
            deprecation_threshold=self.skill_evolution_deprecation_threshold,
        )
```

- [ ] **Step 3: Verify no import errors**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -c "from metaclaw.config import MetaClawConfig; c = MetaClawConfig(); print(c.skill_evolution_config())"`
Expected: Prints `SkillEvolutionConfig(sources=['conversation', 'feedback'], ...)`

- [ ] **Step 4: Commit**

```bash
git add metaclaw/config.py
git commit -m "feat: add skill_evolution_* config fields to MetaClawConfig"
```

---

### Task 9: Wire into api_server.py and trainer.py

**Files:**
- Modify: `metaclaw/api_server.py`
- Modify: `metaclaw/trainer.py`

This task wires all the new components together. It does NOT require tests for the wiring itself (the integration is async server code that's hard to unit-test), but the individual components are fully tested in Tasks 1–7.

- [ ] **Step 1: Add signal detection hook to api_server.py**

In `metaclaw/api_server.py`, add imports at the top:

```python
from .conversation_signal_detector import ConversationSignalDetector
from .feedback_collector import FeedbackCollector
from .signal_aggregator import SignalAggregator
from .skill_signal import SkillAction
```

Add constructor parameters to `MetaClawAPIServer.__init__()` for the new components (all optional, default `None`):

```python
    signal_detector: Optional[ConversationSignalDetector] = None,
    signal_aggregator: Optional[SignalAggregator] = None,
    feedback_collector: Optional[FeedbackCollector] = None,
```

Store as `self._signal_detector`, `self._signal_aggregator`, `self._feedback_collector`.

- [ ] **Step 2: Remove old session-based evolution path**

Remove the old `_session_turns` / `_evolve_skills_for_session()` path to avoid duplicate skill generation:
- Remove `self._session_turns: dict[str, list] = {}` from `__init__()` (line ~452)
- Remove the append to `_session_turns` in the request handler (lines ~1085, ~1141)
- Remove or comment out `_close_session()`'s call to `_evolve_skills_for_session()` (lines ~762–764)
- Remove or deprecate the `_evolve_skills_for_session()` method (line ~1412)

All skill evolution now flows through `SignalAggregator`.

- [ ] **Step 3: Add signal detection after response streaming**

In the response handler (after the assistant response is fully captured and before the turn is finalized), add:

```python
if self._signal_detector and self._signal_aggregator:
    turn_data = {
        "session_id": session_id,
        "turn_num": turn_count,
        "user_message": user_message_text,
        "assistant_response": full_response_text,
        "active_skills": [s["name"] for s in injected_skills] if injected_skills else [],
    }
    signals = self._signal_detector.detect(turn_data)
    if signals:
        self._signal_aggregator.add(signals)
```

- [ ] **Step 4: Add feedback REST endpoints**

Register routes on the FastAPI app in `_build_app()` or wherever routes are defined:

```python
if self._feedback_collector and self._signal_aggregator:
    @app.post("/feedback/rate")
    async def feedback_rate(request: Request):
        body = await request.json()
        sig = self._feedback_collector.rate(
            skill_names=body["skill_names"],
            positive=body["positive"],
            session_id=body.get("session_id", ""),
        )
        self._signal_aggregator.add([sig])
        # Also adjust confidence immediately
        delta = 0.1 if body["positive"] else -0.15
        if self._skill_manager:
            for name in body["skill_names"]:
                self._skill_manager.adjust_confidence(name, delta)
        return JSONResponse({"status": "ok"})

    @app.post("/feedback/correct")
    async def feedback_correct(request: Request):
        body = await request.json()
        sig = self._feedback_collector.correct(
            skill_name=body["skill_name"],
            correction_text=body["correction"],
            session_id=body.get("session_id", ""),
        )
        self._signal_aggregator.add([sig])
        return JSONResponse({"status": "ok"})

    @app.get("/feedback/review")
    async def feedback_review_get(request: Request):
        limit = int(request.query_params.get("limit", "10"))
        candidates = self._feedback_collector.get_review_candidates(limit)
        return JSONResponse({"skills": candidates})

    @app.post("/feedback/review")
    async def feedback_review_post(request: Request):
        body = await request.json()
        signals = self._feedback_collector.submit_review(body["reviews"])
        self._signal_aggregator.add(signals)
        return JSONResponse({"status": "ok", "signals": len(signals)})
```

- [ ] **Step 5: Add `_apply_skill_actions()` and signal aggregator wiring to trainer.py**

In `metaclaw/trainer.py`, add imports:

```python
from .signal_aggregator import SignalAggregator
from .conversation_signal_detector import ConversationSignalDetector
from .feedback_collector import FeedbackCollector
from .skill_signal import SkillAction
```

In `MetaClawTrainer.__init__()`, add:

```python
self._signal_aggregator: Optional[SignalAggregator] = None
self._signal_detector: Optional[ConversationSignalDetector] = None
self._feedback_collector: Optional[FeedbackCollector] = None
```

In `setup()` (after `SkillManager` and `SkillEvolver` are created), add:

```python
if self.config.enable_skill_evolution:
    evo_config = self.config.skill_evolution_config()
    self._signal_aggregator = SignalAggregator(evo_config)
    if "conversation" in evo_config.sources:
        self._signal_detector = ConversationSignalDetector(
            use_llm_detection=evo_config.use_llm_detection,
        )
    if "feedback" in evo_config.sources:
        self._feedback_collector = FeedbackCollector(self.skill_manager)
```

Pass these to the API server constructor.

**CRITICAL:** Update the existing `_maybe_evolve_skills()` method in `trainer.py` (lines ~336–363). The old code calls `self.skill_evolver.evolve(failed_samples, ...)` with `ConversationSample` objects — but after Task 7, `evolve()` expects `list[SkillSignal]`. Replace the body of `_maybe_evolve_skills()` to route through `_signal_aggregator` instead:

```python
async def _maybe_evolve_skills(self, batch):
    """Skill evolution now routes through SignalAggregator."""
    if not self.skill_evolver or not self._signal_aggregator:
        return
    signals = self._signal_aggregator.consume()
    if not signals:
        return
    actions = await self.skill_evolver.evolve(signals, self.skill_manager.skills)
    self._apply_skill_actions(actions)
```

Add `_apply_skill_actions()` method (this is the sole owner of `generation` increments for the signal-based path):

```python
def _apply_skill_actions(self, actions: list[SkillAction]) -> None:
    """Apply SkillEvolver output to the SkillManager."""
    for action in actions:
        if action.action == "create":
            self.skill_manager.add_skill(action.skill)
        elif action.action == "update":
            self.skill_manager.update_skill(action.skill["name"], action.skill)
        elif action.action == "deprecate":
            self.skill_manager.deprecate_skill(action.skill["name"], action.reasoning)
        elif action.action == "link":
            self.skill_manager.link_skills(action.parent_skill, action.skill["name"])
    if any(a.action in ("create", "update") for a in actions):
        self.skill_manager.generation += 1
```

- [ ] **Step 6: Verify imports work**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -c "from metaclaw.trainer import MetaClawTrainer; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Run all tests**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/ -v --ignore=tests/test_v03_live_tinker.py`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add metaclaw/api_server.py metaclaw/trainer.py
git commit -m "feat: wire signal detection and feedback endpoints into api_server and trainer"
```

---

### Task 10: Final integration test and push

- [ ] **Step 1: Run full test suite**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && python -m pytest tests/ -v --ignore=tests/test_v03_live_tinker.py`
Expected: All tests PASS, no regressions

- [ ] **Step 2: Verify all new files are tracked**

Run: `cd /home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw && git status`
Expected: Clean working tree (all committed)

- [ ] **Step 3: Push to origin**

```bash
git push origin kaix/conversation-driven-skill-evolution
```
