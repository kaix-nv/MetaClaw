# Conversation-Driven Skill Evolution for MetaClaw

**Date:** 2026-03-20
**Status:** Draft
**Author:** Kaix + Claude

## Problem Statement

Today, MetaClaw's skill evolution depends on:
1. **PRM scoring** — an LLM judge rates each response, failed samples trigger `SkillEvolver`
2. **RL weight updates** — GRPO/OPD training on the underlying LLM via Tinker/MinT

Both require dataset collection and model training. The goal is to enable **agents that improve purely through conversation** — skills are extracted from user interactions and refined through user feedback, with no dataset curation or weight updates required.

## Design Goals

- Extract skills directly from user conversations (explicit and implicit signals)
- Iterate skills through layered user feedback (binary, corrections, periodic review)
- Support hierarchical skills (broad behavioral → narrow procedural sub-skills)
- Work across any agent type (coding, QA, memory, general-purpose)
- Build as a MetaClaw extension, backward-compatible with existing PRM/RL flow
- Store skills as local markdown files (existing format, git-versionable)

## Approach: Unified SkillEvolver with Pluggable Signal Sources

Refactor `SkillEvolver` to accept multiple signal sources through a common interface. Each source (conversation extraction, user feedback, PRM scores) produces `SkillSignal` objects. The evolver aggregates signals and decides when and how to evolve. This preserves MetaClaw's existing PRM flow as one signal source among several.

**Operating modes:**
- `skills_only` mode: conversation + feedback signals only (no GPU, no RL)
- `rl` mode: adds PRM signal source on top
- `madmax` mode: RL updates scheduled during idle windows, skill evolution always active

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Personal Agent                     │
│            (OpenClaw / CoPaw / etc.)                │
└──────────────────────┬──────────────────────────────┘
                       │ OpenAI-compatible API
                       ▼
┌─────────────────────────────────────────────────────┐
│               MetaClaw API Server                    │
│                  (FastAPI proxy)                     │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │        ConversationSignalDetector            │    │
│  │  • Explicit: "remember this", "always do X" │    │
│  │  • Implicit: corrections, praise, patterns  │    │
│  └──────────────────┬──────────────────────────┘    │
│                     │ SkillSignal[]                  │
│  ┌──────────────────▼──────────────────────────┐    │
│  │           SignalAggregator                   │    │
│  │  • Buffers signals across turns/sessions    │    │
│  │  • Clusters related signals                 │    │
│  │  • Triggers evolution when threshold met    │    │
│  └──────────────────┬──────────────────────────┘    │
│                     │                               │
│  ┌──────────────────▼──────────────────────────┐    │
│  │         SkillEvolver (refactored)            │    │
│  │  • Consumes SkillSignal[] (not samples)     │    │
│  │  • LLM analysis → skill CRUD               │    │
│  │  • Hierarchical skill generation            │    │
│  └──────────────────┬──────────────────────────┘    │
│                     │                               │
│  ┌──────────────────▼──────────────────────────┐    │
│  │      SkillManager (extended)                 │    │
│  │  • add / update / deprecate / link skills   │    │
│  │  • Hierarchical retrieval                   │    │
│  │  • Confidence tracking per skill            │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │       FeedbackCollector (new endpoints)      │    │
│  │  • POST /feedback/rate (thumbs up/down)     │    │
│  │  • POST /feedback/correct (user edits)      │    │
│  │  • GET/POST /feedback/review (batch review) │    │
│  └──────────────────┬──────────────────────────┘    │
│                     │ SkillSignal[]                  │
│                     └──► SignalAggregator            │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │       PRMSignalAdapter (backward compat)     │    │
│  │  • Wraps PRMScorer output → SkillSignal     │    │
│  └──────────────────┬──────────────────────────┘    │
│                     │ SkillSignal[]                  │
│                     └──► SignalAggregator            │
└─────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. SkillSignal — The Core Abstraction

A normalized event that any source can produce. All signal sources emit these; the evolver consumes only these.

```python
@dataclass
class SkillSignal:
    source: str          # "conversation", "feedback", "prm"
    signal_type: str     # See signal type taxonomy below
    content: dict        # Source-specific payload
    confidence: float    # 0.0–1.0, used for prioritization
    timestamp: str       # ISO 8601
    session_id: str      # Links back to conversation session
```

**Signal type taxonomy:**

| Source | Signal Type | Trigger | Content Payload |
|--------|------------|---------|-----------------|
| conversation | `explicit_save` | "remember this", "save as skill", "from now on..." | `{user_message, context, response}` |
| conversation | `implicit_correction` | "no, do X instead", "that's wrong, you should..." | `{user_message, wrong_response, correction}` |
| conversation | `implicit_success` | User accepts novel approach on non-trivial task | `{prompt, response, task_complexity}` |
| conversation | `pattern_detected` | Same correction given 2+ times across sessions | `{pattern, examples[], frequency}` |
| feedback | `thumbs_up` | Binary positive rating | `{skill_names[], response_id}` |
| feedback | `thumbs_down` | Binary negative rating | `{skill_names[], response_id}` |
| feedback | `user_correction` | User provides corrected skill text | `{skill_name, original, corrected}` |
| feedback | `review_approved` | User approves skill in periodic review | `{skill_name}` |
| feedback | `review_rejected` | User rejects skill in periodic review | `{skill_name, reason}` |
| prm | `prm_failure` | PRM reward <= 0 | `{sample, reward, active_skills[]}` |
| prm | `prm_success` | PRM reward > 0 | `{sample, reward, active_skills[]}` |

### 2. ConversationSignalDetector

Runs on each conversation turn inside `api_server.py`, post-response. Analyzes the user's next message (if available) to determine if the previous response triggered a signal.

**File:** `metaclaw/conversation_signal_detector.py`

```python
class ConversationSignalDetector:
    """Detects skill-worthy signals from conversation turns."""

    def __init__(self, llm_client=None, use_llm_detection: bool = True):
        self._llm_client = llm_client
        self._use_llm = use_llm_detection
        self._correction_history: dict[str, list] = {}  # session_id → corrections

    def detect(self, turn_data: dict) -> list[SkillSignal]:
        """
        Analyze a conversation turn for skill signals.

        turn_data keys:
            session_id, turn_num, user_message, assistant_response,
            previous_user_message (if available), active_skills[]
        """
        signals = []
        signals.extend(self._detect_explicit(turn_data))
        signals.extend(self._detect_implicit(turn_data))
        return signals
```

**Explicit detection** — keyword/pattern matching (fast, no LLM needed):
- "remember this" / "save this" / "from now on" / "always do X" / "never do X"
- "that's a good approach, keep doing that"
- Direct skill management: "create a skill for..." / "add this to your skills"

**Implicit detection** — two-tier:
1. **Fast heuristic pass:** Detect correction patterns ("no", "wrong", "instead", "actually") in the user's follow-up message. High recall, moderate precision.
2. **LLM confirmation pass (optional):** For heuristic-flagged turns, ask LLM to confirm whether this is genuinely a correction and extract the skill-worthy content. Reduces false positives. Configurable via `use_llm_detection`.

**Cross-session pattern detection:**
- Maintains a sliding window of recent corrections per session
- When the same correction appears 2+ times (clustered by embedding similarity), emits a `pattern_detected` signal with higher confidence

### 3. FeedbackCollector

New REST endpoints added to MetaClaw's FastAPI server for structured feedback from agent UIs.

**File:** `metaclaw/feedback_collector.py`

```python
class FeedbackCollector:
    """Collects structured user feedback and emits SkillSignals."""

    def __init__(self, skill_manager: SkillManager):
        self._skill_manager = skill_manager
        self._feedback_buffer: list[SkillSignal] = []

    def rate(self, skill_names: list[str], positive: bool,
             session_id: str, response_id: str = "") -> SkillSignal: ...

    def correct(self, skill_name: str, correction_text: str,
                session_id: str) -> SkillSignal:
        """Create a user_correction signal.

        Looks up the current skill content from SkillManager via
        get_skill(skill_name) to populate the 'original' field in the
        signal payload, so the evolver can see what changed.
        """
        original = self._skill_manager.get_skill(skill_name)
        return SkillSignal(
            source="feedback",
            signal_type="user_correction",
            content={
                "skill_name": skill_name,
                "original": original.get("content", "") if original else "",
                "corrected": correction_text,
            },
            ...
        )

    def get_review_candidates(self, limit: int = 10) -> list[dict]:
        """Return skills sorted by staleness or low confidence for user review."""
        ...

    def submit_review(self, reviews: list[dict]) -> list[SkillSignal]:
        """Process batch review results: approve/edit/reject per skill."""
        ...
```

**Note:** `SkillManager.get_skill(name)` is a new method (see SkillManager Extensions below) that returns the full skill dict by name, or `None` if not found.

**API endpoints** (added to `api_server.py`):

```
POST /feedback/rate
  Body: {"skill_names": [...], "positive": bool, "session_id": "..."}

POST /feedback/correct
  Body: {"skill_name": "...", "correction": "...", "session_id": "..."}

GET  /feedback/review?limit=10
  Returns: skills needing review, sorted by staleness/low confidence

POST /feedback/review
  Body: {"reviews": [{"skill_name": "...", "action": "approve|edit|reject", "edited_content": "..."}]}
```

These endpoints are optional — the system works without them. They enable richer feedback from agents/UIs that support structured feedback channels.

### 4. PRMSignalAdapter

Wraps existing `PRMScorer` output into `SkillSignal` format for backward compatibility. The existing PRM-driven evolution path becomes just another signal source.

**File:** `metaclaw/prm_signal_adapter.py`

```python
class PRMSignalAdapter:
    """Adapts PRMScorer results to SkillSignal format.

    Only adapts samples with reward != 0 (clear signal). Samples with
    reward=0 (ambiguous) are skipped — they provide no actionable signal
    for skill evolution.
    """

    # Confidence mapping: clear failures/successes get high confidence.
    # reward=0 is filtered out before reaching this method.
    _CONFIDENCE_MAP = {-1.0: 1.0, 1.0: 1.0}

    def should_adapt(self, sample: ConversationSample) -> bool:
        """Return False for ambiguous samples (reward=0)."""
        return sample.reward != 0.0

    def adapt(self, sample: ConversationSample,
              active_skills: list[str]) -> SkillSignal:
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
            timestamp=datetime.now().isoformat(),
            session_id=sample.session_id,
        )
```

### 5. SignalAggregator

Buffers signals from all sources, clusters related signals, and decides when to trigger skill evolution.

**File:** `metaclaw/signal_aggregator.py`

```python
class SignalAggregator:
    """Aggregates SkillSignals and triggers evolution.

    Thread/async safety: add() and consume() may be called from different
    coroutines on the same event loop. consume() uses atomic swap to prevent
    signal loss if add() interleaves.

    Signal source gating: only signals whose source is listed in
    config.sources are accepted. This is how skills_only mode excludes
    PRM signals and how rl mode enables them.
    """

    def __init__(self, config: SkillEvolutionConfig):
        self._buffer: list[SkillSignal] = []
        self._config = config
        self._allowed_sources: set[str] = set(config.sources)

    def add(self, signals: list[SkillSignal]) -> None:
        """Add signals, filtering out sources not in config.sources."""
        self._buffer.extend(
            s for s in signals if s.source in self._allowed_sources
        )

    def should_evolve(self) -> bool:
        """Check if accumulated signals warrant evolution."""
        # Trigger conditions (any one sufficient):
        # 1. Any explicit_save signal (immediate)
        # 2. Accumulated corrections exceed threshold (e.g., 3+)
        # 3. Pattern detected with high confidence
        # 4. PRM failure rate below threshold (existing behavior)
        ...

    def consume(self) -> list[SkillSignal]:
        """Return and clear buffered signals for evolution.

        Uses atomic swap (single assignment) to prevent signal loss
        if another coroutine calls add() between read and clear.
        """
        signals, self._buffer = self._buffer, []
        return signals
```

**Trigger conditions (configurable):**

| Condition | Default | Behavior |
|-----------|---------|----------|
| Explicit save signal | Immediate | Any `explicit_save` triggers evolution right away |
| Correction accumulation | 3 corrections | After N corrections buffered, trigger evolution |
| Pattern detection | confidence > 0.8 | High-confidence cross-session patterns trigger immediately |
| PRM failure rate | < 0.4 | Existing threshold-based trigger (when PRM enabled) |
| Time-based | 30 minutes | Flush buffer periodically even if thresholds not met |

### 6. SkillEvolver (Refactored)

The existing `SkillEvolver` is refactored to consume `SkillSignal[]` instead of `ConversationSample[]`. The LLM analysis prompt adapts based on which signal sources are present.

**Key changes to `skill_evolver.py`:**

```python
class SkillEvolver:
    async def evolve(
        self,
        signals: list[SkillSignal],           # Was: failed_samples
        current_skills: dict[str, Any],        # Same shape: {general_skills, task_specific_skills, common_mistakes}
    ) -> list[SkillAction]:                    # Was: list[dict]
        """
        Analyze signals and propose skill actions.

        current_skills accepts the existing SkillManager.skills dict
        shape ({general_skills: [], task_specific_skills: {}, common_mistakes: []}).
        The prompt builder enumerates existing skills from this dict for
        deduplication, same as the original _build_analysis_prompt().

        Returns SkillAction objects (not raw dicts) that specify:
        - create: new skill
        - update: modify existing skill
        - deprecate: mark skill as low-confidence
        - link: create parent-child relationship
        """
        prompt = self._build_signal_analysis_prompt(signals, current_skills)
        response = await asyncio.to_thread(self._call_llm, prompt)
        return self._parse_actions_response(response)

    def _build_signal_analysis_prompt(
        self,
        signals: list[SkillSignal],
        current_skills: dict[str, Any],  # {general_skills, task_specific_skills, common_mistakes}
    ) -> str:
        """Build LLM prompt from signals + existing skill inventory.

        Enumerates existing skills from current_skills dict (same structure
        as SkillManager.skills) for deduplication. Groups signals by type
        and formats each group with type-appropriate context.
        """
        ...
```

**SkillAction** — the output of evolution:

```python
@dataclass
class SkillAction:
    action: str              # "create", "update", "deprecate", "link"
    skill: dict              # Skill dict (name, description, content, category)
    parent_skill: str = ""   # For "link" action: parent skill name
    reasoning: str = ""      # LLM's reasoning for this action
    source_signals: list[str] = field(default_factory=list)  # Signal IDs that motivated this
```

**Prompt strategy by signal type:**

| Signal Mix | Analysis Focus |
|------------|---------------|
| Mostly explicit saves | Extract skill directly from user's description, minimal LLM interpretation |
| Mostly corrections | Identify what went wrong, generate skill that prevents the mistake |
| Mostly patterns | Synthesize recurring theme into a general behavioral skill |
| Mostly PRM failures | Existing failure analysis (cluster → diagnose → propose) |
| Mixed | Weighted analysis — explicit/correction signals get priority over PRM |

### 7. SkillManager Extensions

Extend the existing `SkillManager` with new methods and extended frontmatter parsing.

**a) `_parse_skill_md()` changes (REQUIRED):**

The existing `_parse_skill_md()` in `skill_manager.py` only extracts `name`, `description`, and `category` from frontmatter. It must be extended to also parse: `parent`, `children`, `confidence`, `provenance`, `created_from_session`, `deprecated`. Unknown keys are preserved in a `frontmatter` dict for forward compatibility.

```python
def _parse_skill_md(path: str) -> Optional[Dict[str, Any]]:
    """Parse SKILL.md — now extracts all frontmatter fields."""
    # ... existing parsing ...
    return {
        "name": name,
        "description": description,
        "category": category,
        "content": body,
        # New fields (with defaults for backward compat):
        "parent": fm.get("parent", None),
        "children": _parse_list(fm.get("children", "")),  # "a, b, c" → ["a", "b", "c"]
        "confidence": float(fm.get("confidence", 0.6)),     # default = initial_confidence
        "provenance": fm.get("provenance", "manual"),
        "created_from_session": fm.get("created_from_session", ""),
        "deprecated": fm.get("deprecated", "false").lower() == "true",
    }
```

Similarly, `_write_skill_md()` must be extended to write these new fields back to frontmatter.

**b) New lookup method:**
```python
def get_skill(self, name: str) -> Optional[dict]:
    """Return the full skill dict by name, or None if not found."""
    for skill in self._iter_all_skills():
        if skill.get("name") == name:
            return skill
    return None

def _iter_all_skills(self) -> Iterator[dict]:
    """Iterate over all skills across all categories."""
    yield from self.skills.get("general_skills", [])
    for cat_skills in self.skills.get("task_specific_skills", {}).values():
        yield from cat_skills
    yield from self.skills.get("common_mistakes", [])
```

**c) Skill update and deprecation:**
```python
def update_skill(self, name: str, updates: dict) -> bool:
    """Update an existing skill's description, content, or category.
    Writes changes to disk immediately."""
    ...

def deprecate_skill(self, name: str, reason: str = "") -> bool:
    """Mark a skill as deprecated. Sets deprecated=True in frontmatter.
    Deprecated skills are excluded from retrieval but not deleted."""
    ...
```

**d) Hierarchical skill support:**

Extended frontmatter format:
```yaml
---
name: debug-systematically
description: Use when diagnosing a bug...
category: coding
parent: null
children: [check-logs-first, reproduce-before-fixing, bisect-to-root-cause]
confidence: 0.85
provenance: conversation  # "conversation", "prm", "feedback", "manual"
created_from_session: abc-123
deprecated: false
---
```

```python
def link_skills(self, parent_name: str, child_name: str) -> bool:
    """Create parent-child relationship between skills.
    Updates both parent's children list and child's parent field.
    Writes both skill files to disk."""
    ...

def get_skill_tree(self, name: str) -> dict:
    """Return skill with its full sub-skill tree (recursive)."""
    ...
```

**e) Confidence tracking:**

Each skill tracks a confidence score (0.0–1.0) stored in frontmatter, adjusted based on feedback:
- Thumbs up: `confidence = min(1.0, confidence + 0.1)`
- Thumbs down: `confidence = max(0.0, confidence - 0.15)`
- User correction: `confidence = 0.5` (reset after edit)
- Review approved: `confidence = min(1.0, confidence + 0.2)`
- Review rejected: skill deprecated

```python
def adjust_confidence(self, name: str, delta: float) -> None:
    """Adjust a skill's confidence score and persist to disk."""
    ...
```

**f) Retrieval changes:**

Skills with `confidence < deprecation_threshold` (default 0.3) are excluded from retrieval. Deprecated skills are never retrieved. When retrieving a parent skill, its children are included automatically.

The existing `retrieve()` method is modified to filter by confidence and expand parent→children before returning.

---

## Integration Points in Existing MetaClaw

### Component wiring

`SignalAggregator` is a shared singleton instantiated in `MetaClawTrainer.setup()` (or in `MetaClawLauncher` for `skills_only` mode) and passed to both the API server and the trainer:

```python
# In trainer.py setup() or launcher.py for skills_only mode:
self._signal_aggregator = SignalAggregator(config.skill_evolution)
self._signal_detector = ConversationSignalDetector(
    use_llm_detection=config.skill_evolution.use_llm_detection
)
self._feedback_collector = FeedbackCollector(self.skill_manager)

# Pass to API server at construction:
api_server = MetaClawAPIServer(
    ...,
    signal_detector=self._signal_detector,
    signal_aggregator=self._signal_aggregator,
    feedback_collector=self._feedback_collector,
)
```

`SignalAggregator` is safe to share because both the API server handlers and the trainer run on the same asyncio event loop, and `consume()` uses atomic swap (see SignalAggregator section).

### api_server.py changes

The new per-turn signal detection **replaces** the existing `_session_turns` / `_evolve_skills_for_session()` end-of-session path. The old path is removed to avoid duplicate skill generation from the same session data.

```python
# In the response handler, after streaming response back:
if self._signal_detector:
    turn_data = {
        "session_id": session_id,
        "turn_num": turn_num,
        "user_message": user_message,
        "assistant_response": full_response,
        "active_skills": [s["name"] for s in injected_skills],
    }
    signals = self._signal_detector.detect(turn_data)
    if signals:
        self._signal_aggregator.add(signals)

    if self._signal_aggregator.should_evolve():
        # Non-blocking evolution
        asyncio.create_task(self._evolve_from_signals())

# REMOVED: _session_turns buffering and _evolve_skills_for_session()
# All skill evolution now flows through SignalAggregator.
```

### trainer.py changes

`_apply_skill_actions()` is a new method on `MetaClawTrainer` that maps `SkillAction` objects to `SkillManager` operations:

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

# In the training loop, skill evolution step:
if self.skill_evolver and self._signal_aggregator:
    signals = self._signal_aggregator.consume()
    if signals:
        actions = await self.skill_evolver.evolve(signals, self.skill_manager.skills)
        self._apply_skill_actions(actions)
```

### config.py additions

```python
@dataclass
class SkillEvolutionConfig:
    sources: list[str] = field(default_factory=lambda: ["conversation", "feedback"])
    # Trigger thresholds
    correction_threshold: int = 3
    pattern_confidence_threshold: float = 0.8
    prm_failure_threshold: float = 0.4
    flush_interval_minutes: int = 30
    # LLM detection
    use_llm_detection: bool = True
    detection_model: str = ""  # Falls back to SKILL_EVOLVER_MODEL
    # Hierarchy
    enable_hierarchy: bool = True
    max_depth: int = 3
    # Confidence
    initial_confidence: float = 0.6
    deprecation_threshold: float = 0.3
```

**How `sources` controls component instantiation:**

The `sources` field has two enforcement points:

1. **Component instantiation** (in `trainer.setup()` or `launcher.py`):
   - `"conversation" in sources` → create `ConversationSignalDetector`, pass to API server
   - `"feedback" in sources` → create `FeedbackCollector`, register feedback endpoints
   - `"prm" in sources` → create `PRMSignalAdapter`, wire to existing `PRMScorer`
   - If a source is not listed, its component is not created (no overhead)

2. **Signal filtering** (in `SignalAggregator.add()`):
   - Signals whose `source` is not in `config.sources` are dropped
   - This is a safety net in case signals arrive from an unexpected path

**Mode defaults:**
- `skills_only`: `sources = ["conversation", "feedback"]` (no PRM, no RL)
- `rl`: `sources = ["conversation", "feedback", "prm"]`
- `madmax`: same as `rl`, with scheduler wrapping the training loop

---

## Data Flow

### Flow 1: Explicit skill extraction (no training)

```
User: "From now on, when I ask about meetings, always check my calendar first"
  │
  ▼
ConversationSignalDetector.detect()
  → SkillSignal(source="conversation", signal_type="explicit_save",
                content={user_message: "From now on...", context: ...},
                confidence=0.95)
  │
  ▼
SignalAggregator.add() → should_evolve() = True (explicit = immediate)
  │
  ▼
SkillEvolver.evolve([signal], current_skills)
  → LLM extracts: SkillAction(action="create",
      skill={name: "check-calendar-for-meetings",
             description: "When user asks about meetings, check calendar API first",
             content: "## Check Calendar Before Answering Meeting Questions\n...",
             category: "productivity"})
  │
  ▼
SkillManager.add_skill() → writes SKILL.md to disk
  │
  ▼
Next conversation: skill injected into system prompt
```

### Flow 2: Implicit correction → skill refinement

```
Turn 1: Agent gives verbose debug output
User: "Just show me the error line, not the full trace"
  │
  ▼
ConversationSignalDetector.detect()
  → SkillSignal(signal_type="implicit_correction",
                content={wrong_response: "...(full trace)...",
                         correction: "Just show me the error line"},
                confidence=0.7)
  │
  ▼
SignalAggregator: buffered (1 correction, threshold=3)

... 2 more similar corrections across sessions ...

SignalAggregator: threshold met → should_evolve() = True
  │
  ▼
SkillEvolver.evolve([3 correction signals], current_skills)
  → LLM: SkillAction(action="create",
      skill={name: "concise-error-output",
             description: "When showing errors, display only the relevant line...",
             content: "..."})
  OR if existing skill "debug-systematically" covers this:
  → SkillAction(action="update", skill={name: "debug-systematically", ...})
  → SkillAction(action="link", parent="debug-systematically",
                child="concise-error-output")
```

### Flow 3: Feedback-driven refinement

```
User clicks thumbs-down on response where "structured-reasoning" skill was active
  │
  ▼
POST /feedback/rate {skill_names: ["structured-reasoning"], positive: false}
  │
  ▼
FeedbackCollector.rate() → SkillSignal(signal_type="thumbs_down")
  │
  ▼
SkillManager: structured-reasoning.confidence -= 0.15

... skill appears in periodic review ...

GET /feedback/review → returns "structured-reasoning" (low confidence)
User edits the skill content
POST /feedback/review {reviews: [{skill_name: "structured-reasoning",
                                   action: "edit", edited_content: "..."}]}
  │
  ▼
SkillManager.update_skill("structured-reasoning", {content: edited_content})
  → confidence reset to 0.5
```

### Flow 4: PRM-driven evolution (backward compatible)

```
PRMScorer rates response as -1
  │
  ▼
PRMSignalAdapter.adapt(sample) → SkillSignal(source="prm", signal_type="prm_failure")
  │
  ▼
SignalAggregator: accumulates PRM failures
  → when failure_rate < 0.4: should_evolve() = True
  │
  ▼
SkillEvolver.evolve() — same as existing behavior, just via signal interface
```

---

## File Structure (new/modified files)

```
metaclaw/
├── skill_signal.py              [NEW]  SkillSignal, SkillAction dataclasses
├── conversation_signal_detector.py  [NEW]  Explicit + implicit detection
├── feedback_collector.py        [NEW]  Feedback endpoints + signal emission
├── signal_aggregator.py         [NEW]  Buffer, cluster, trigger logic
├── prm_signal_adapter.py        [NEW]  Wraps PRMScorer → SkillSignal
├── skill_evolver.py             [MOD]  Refactored to consume SkillSignal[]
├── skill_manager.py             [MOD]  +update, +deprecate, +hierarchy, +confidence
├── api_server.py                [MOD]  +signal detector hook, +feedback endpoints
├── trainer.py                   [MOD]  +signal aggregator integration
├── config.py                    [MOD]  +SkillEvolutionConfig
```

---

## Hierarchical Skill Model

Skills form a tree. Broad skills decompose into narrow sub-skills:

```
debug-systematically (broad, behavioral)
├── check-logs-first (narrow, procedural)
├── reproduce-before-fixing (narrow, procedural)
└── concise-error-output (narrow, procedural)

manage-meetings (broad, behavioral)
├── check-calendar-for-meetings (narrow, procedural)
└── summarize-meeting-action-items (narrow, procedural)
```

**Rules:**
- Max depth: 3 levels (configurable)
- A parent skill's content is a high-level guide; children contain specific procedures
- When a parent is retrieved, all children are included in injection
- Children can exist independently (orphan skills get adopted when patterns emerge)
- SkillEvolver proposes `link` actions when it detects a new skill fits under an existing parent

**Storage:** Parent/child references stored in YAML frontmatter. Skill files remain independent (no nesting on filesystem).

---

## Open Questions

1. **Feedback flow:** Should feedback come through in-conversation natural language only (ConversationSignalDetector handles everything), separate API endpoints only, or both? Current design supports both — the endpoints are optional.

2. **Skill conflict resolution:** When two skills give contradictory guidance, which wins? Options: higher confidence, more recent, user-pinned priority. Needs decision.

3. **Skill garbage collection:** How aggressively should deprecated skills be cleaned up? Options: never delete (just stop retrieving), delete after N days below threshold, user-only deletion.

4. **Cross-agent skill sharing:** If OpenClaw and CoPaw both connect through MetaClaw, should they share the same skill bank or have agent-specific skills? Current design: shared bank (single `skills/` directory).

5. **Agent self-reflection (future):** Should the agent review its own performance post-conversation and generate feedback signals autonomously? Deferred for v1 but the signal architecture supports it trivially (add a `SelfReflectionDetector` signal source).
