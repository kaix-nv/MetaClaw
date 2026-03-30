"""Parse OpenCode agent traces to extract error→fix pairs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SOLUTION_PATHS = {"solution.py", "/testbed/solution.py"}
_FAIL_PATTERNS = re.compile(r"FAIL|Error|error|Traceback|AssertionError|RuntimeError|ImportError|TypeError|ValueError")
_PASS_PATTERNS = re.compile(r"PASSED|passed|OK\b")
_TEST_COMMANDS = re.compile(r"pytest|python.*test|PYTHONPATH.*test")


class TraceParser:
    """Extract error→fix pairs from OpenCode agent traces."""

    def parse(self, trace_text: str) -> list[dict]:
        """Parse a trace (newline-delimited JSON) into error→fix pairs.

        An error→fix pair is: write(code) → run(test) → FAIL → [reasoning] → write(fix) → run(test) → PASS/FAIL
        Only consecutive error→fix sequences are captured.
        """
        if not trace_text.strip():
            return []

        entries = []
        for line in trace_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        # Extract events: writes to solution.py, test runs (bash), and text (reasoning)
        events = []
        for entry in entries:
            typ = entry.get("type", "")
            part = entry.get("part", {})

            if typ == "tool_use":
                tool = part.get("tool", "")
                state = part.get("state", {})
                inp = state.get("input", {})
                out = state.get("output", "")

                if tool == "write":
                    path = inp.get("filePath", "")
                    # Check if writing to solution file
                    if any(path.endswith(sp) for sp in _SOLUTION_PATHS) or "solution" in path:
                        events.append({
                            "type": "write",
                            "content": inp.get("content", ""),
                            "path": path,
                        })

                elif tool == "bash":
                    cmd = inp.get("command", "")
                    if _TEST_COMMANDS.search(cmd):
                        is_fail = bool(_FAIL_PATTERNS.search(out)) and not bool(_PASS_PATTERNS.search(out))
                        is_pass = bool(_PASS_PATTERNS.search(out)) and not bool(re.search(r"FAILED|failed", out))
                        events.append({
                            "type": "test_fail" if is_fail else ("test_pass" if is_pass else "test_other"),
                            "command": cmd,
                            "output": out,
                        })

            elif typ == "text":
                text = part.get("text", "").strip()
                if text:
                    events.append({"type": "reasoning", "text": text})

        # Find error→fix sequences
        return self._extract_pairs(events)

    def _extract_pairs(self, events: list[dict]) -> list[dict]:
        """Find sequences: write → test_fail → [reasoning] → write → test_*"""
        pairs = []
        i = 0
        while i < len(events):
            # Look for: write → test_fail
            if (events[i]["type"] == "write"
                    and i + 1 < len(events)
                    and events[i + 1]["type"] == "test_fail"):

                code_before = events[i]["content"]
                error_message = events[i + 1]["output"]

                # Collect reasoning text
                reasoning_parts = []
                j = i + 2
                while j < len(events) and events[j]["type"] == "reasoning":
                    reasoning_parts.append(events[j]["text"])
                    j += 1

                # Look for next write (the fix)
                if j < len(events) and events[j]["type"] == "write":
                    code_after = events[j]["content"]
                    pairs.append({
                        "error_message": error_message[:1000],
                        "code_before": code_before,
                        "code_after": code_after,
                        "agent_reasoning": " ".join(reasoning_parts),
                    })
                    # Restart from the fix-write so it can begin the next error cycle
                    i = j
                    continue

            i += 1

        return pairs
