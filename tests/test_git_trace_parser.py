# tests/test_git_trace_parser.py
import os
import subprocess
import pytest
from pathlib import Path
from eval.cutile.git_trace_parser import GitTraceParser


@pytest.fixture
def git_workspace(tmp_path):
    """Create a fake git workspace simulating an agent's solve attempts."""
    ws = tmp_path / "workspace"
    ws.mkdir()

    # Init git
    subprocess.run(["git", "init"], cwd=ws, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=ws, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=ws, capture_output=True)

    # Initial commit
    (ws / "problem-spec.yaml").write_text("kernel: softmax\n")
    subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial workspace"], cwd=ws, capture_output=True)

    # Attempt 1: bad code
    (ws / "solution.py").write_text("import cuda.tile as ct\n\ndef bad():\n    pass\n")
    subprocess.run(["git", "add", "solution.py"], cwd=ws, capture_output=True)
    subprocess.run(["git", "commit", "-m", "attempt 1: first try"], cwd=ws, capture_output=True)

    # Attempt 2: fixed code
    (ws / "solution.py").write_text("import cuda.tile as ct\n\n@ct.kernel\ndef good():\n    ct.load()\n")
    subprocess.run(["git", "add", "solution.py"], cwd=ws, capture_output=True)
    subprocess.run(["git", "commit", "-m", "attempt 2: fix kernel decorator"], cwd=ws, capture_output=True)

    # Write results.log with error from attempt 1
    (ws / "results.log").write_text(
        "FAILED: RuntimeError: ct.mma expects 2D tiles\n"
        "PASSED\n1 passed in 5.2s\n"
        "FINAL: PASS attempts=2 kernel=softmax\n"
    )

    return ws


def test_parse_git_history(git_workspace):
    parser = GitTraceParser()
    pairs = parser.parse(str(git_workspace))
    assert len(pairs) >= 1
    pair = pairs[0]
    assert "code_before" in pair
    assert "code_after" in pair
    assert "bad()" in pair["code_before"]
    assert "good()" in pair["code_after"]


def test_parse_extracts_errors(git_workspace):
    parser = GitTraceParser()
    pairs = parser.parse(str(git_workspace))
    assert pairs[0].get("error_message", "")
    assert "ct.mma" in pairs[0]["error_message"] or len(pairs[0]["error_message"]) > 0


def test_parse_detects_pass(git_workspace):
    parser = GitTraceParser()
    result = parser.get_final_status(str(git_workspace))
    assert result["passed"] is True
    assert result["attempts"] == 2


def test_parse_empty_workspace(tmp_path):
    parser = GitTraceParser()
    pairs = parser.parse(str(tmp_path))
    assert pairs == []


def test_parse_no_solution(tmp_path):
    """Workspace with git but no solution.py commits."""
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init"], cwd=ws, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=ws, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=ws, capture_output=True)
    (ws / "readme").write_text("hi")
    subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=ws, capture_output=True)
    parser = GitTraceParser()
    pairs = parser.parse(str(ws))
    assert pairs == []
