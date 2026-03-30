import json
import pytest
from eval.cutile.trace_parser import TraceParser


def _make_trace_lines():
    """Create a realistic trace with write → bash(test) → error → write(fix) → bash(test) → pass."""
    return [
        json.dumps({"type": "step_start", "part": {"type": "step"}}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "write",
            "state": {
                "input": {"filePath": "/testbed/solution.py", "content": "import cuda.tile as ct\n\ndef bad_kernel():\n    pass\n"},
                "output": "",
            }
        }}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "bash",
            "state": {
                "input": {"command": "python -m pytest test/test_softmax.py"},
                "output": "FAILED: RuntimeError: ct.mma expects 2D tiles, got shape (1, 128, 64)",
            }
        }}),
        json.dumps({"type": "text", "part": {"text": "I need to reshape the tile to 2D before MMA."}}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "write",
            "state": {
                "input": {"filePath": "/testbed/solution.py", "content": "import cuda.tile as ct\n\ndef fixed_kernel():\n    tile = ct.reshape(t, (M, K))\n"},
                "output": "",
            }
        }}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "bash",
            "state": {
                "input": {"command": "python -m pytest test/test_softmax.py"},
                "output": "PASSED\n1 passed in 5.2s",
            }
        }}),
    ]


def test_parse_error_fix_pairs():
    parser = TraceParser()
    lines = _make_trace_lines()
    pairs = parser.parse("\n".join(lines))
    assert len(pairs) >= 1
    pair = pairs[0]
    assert "error_message" in pair
    assert "ct.mma expects 2D tiles" in pair["error_message"]
    assert "code_before" in pair
    assert "code_after" in pair
    assert "bad_kernel" in pair["code_before"]
    assert "fixed_kernel" in pair["code_after"]


def test_parse_with_agent_reasoning():
    parser = TraceParser()
    lines = _make_trace_lines()
    pairs = parser.parse("\n".join(lines))
    assert pairs[0].get("agent_reasoning", "")
    assert "reshape" in pairs[0]["agent_reasoning"].lower()


def test_parse_no_errors():
    """Trace where code passes on first try — no error→fix pairs."""
    lines = [
        json.dumps({"type": "tool_use", "part": {
            "tool": "write",
            "state": {"input": {"filePath": "/testbed/solution.py", "content": "good code"}, "output": ""}
        }}),
        json.dumps({"type": "tool_use", "part": {
            "tool": "bash",
            "state": {"input": {"command": "pytest test/"}, "output": "PASSED\n1 passed"}
        }}),
    ]
    parser = TraceParser()
    pairs = parser.parse("\n".join(lines))
    assert len(pairs) == 0


def test_parse_empty_trace():
    parser = TraceParser()
    pairs = parser.parse("")
    assert pairs == []


def test_parse_multiple_errors():
    """Two error→fix cycles in one trace."""
    lines = [
        # First error cycle
        json.dumps({"type": "tool_use", "part": {"tool": "write", "state": {"input": {"filePath": "/testbed/solution.py", "content": "v1"}, "output": ""}}}),
        json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": "pytest"}, "output": "FAILED: ImportError: ct.mma not found"}}}),
        json.dumps({"type": "text", "part": {"text": "Need to use ct.matmul instead"}}),
        json.dumps({"type": "tool_use", "part": {"tool": "write", "state": {"input": {"filePath": "/testbed/solution.py", "content": "v2"}, "output": ""}}}),
        json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": "pytest"}, "output": "FAILED: shape mismatch"}}}),
        json.dumps({"type": "text", "part": {"text": "Need to fix tile shape"}}),
        json.dumps({"type": "tool_use", "part": {"tool": "write", "state": {"input": {"filePath": "/testbed/solution.py", "content": "v3"}, "output": ""}}}),
        json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": "pytest"}, "output": "PASSED"}}}),
    ]
    parser = TraceParser()
    pairs = parser.parse("\n".join(lines))
    assert len(pairs) == 2
    assert "ct.mma not found" in pairs[0]["error_message"]
    assert "shape mismatch" in pairs[1]["error_message"]
