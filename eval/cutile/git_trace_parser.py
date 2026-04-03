"""Parse git history from autoresearch-style solve workspaces into error→fix pairs."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class GitTraceParser:
    """Extract error→fix pairs from git commit history of a solve workspace."""

    def parse(self, workspace_dir: str) -> list[dict]:
        """Parse git history of a workspace to find error→fix pairs.

        Looks for consecutive commits that modify solution.py.
        Pairs each commit with errors from results.log between those commits.
        """
        ws = Path(workspace_dir)
        if not (ws / ".git").exists():
            return []

        # Get list of commits that touch solution.py
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--follow", "--", "solution.py"],
                cwd=ws, capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError:
            return []

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return []

        # Commits are newest-first, reverse for chronological order
        commits = [l.split()[0] for l in reversed(lines)]

        # Read errors from results.log
        errors = self._parse_results_log(ws / "results.log")

        # Extract pairs from consecutive commits
        pairs = []
        for i in range(len(commits) - 1):
            old_sha = commits[i]
            new_sha = commits[i + 1]

            code_before = self._git_show(ws, old_sha, "solution.py")
            code_after = self._git_show(ws, new_sha, "solution.py")

            if not code_before or not code_after:
                continue

            # Find error message between these attempts
            error_msg = errors[i] if i < len(errors) else ""

            # Get commit message for context
            commit_msg = self._git_commit_msg(ws, new_sha)

            pairs.append({
                "error_message": error_msg,
                "code_before": code_before,
                "code_after": code_after,
                "commit_message": commit_msg,
            })

        return pairs

    def get_final_status(self, workspace_dir: str) -> dict:
        """Parse FINAL: line from results.log."""
        results_log = Path(workspace_dir) / "results.log"
        if not results_log.exists():
            return {"passed": False, "attempts": 0}

        content = results_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"FINAL:\s*(PASS|FAIL)\s+attempts=(\d+)", content)
        if match:
            return {
                "passed": match.group(1) == "PASS",
                "attempts": int(match.group(2)),
            }
        # Fallback: check if any PASSED appears
        return {
            "passed": "PASSED" in content or "passed" in content,
            "attempts": 0,
        }

    def _parse_results_log(self, path: Path) -> list[str]:
        """Extract error messages from results.log.

        Returns a list of error messages, one per test run (between attempts).
        """
        if not path.exists():
            return []

        content = path.read_text(encoding="utf-8", errors="replace")
        errors = []
        current_error = []

        for line in content.split("\n"):
            if re.search(r"FAILED|Error|Traceback|AssertionError|RuntimeError|ImportError", line):
                current_error.append(line.strip())
            elif re.search(r"PASSED|passed", line):
                if current_error:
                    errors.append("\n".join(current_error))
                    current_error = []
                errors.append("")  # empty = this attempt passed

        if current_error:
            errors.append("\n".join(current_error))

        return errors

    @staticmethod
    def _git_show(ws: Path, sha: str, filename: str) -> str:
        """Get file content at a specific commit."""
        try:
            result = subprocess.run(
                ["git", "show", f"{sha}:{filename}"],
                cwd=ws, capture_output=True, text=True, check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    @staticmethod
    def _git_commit_msg(ws: Path, sha: str) -> str:
        """Get commit message."""
        try:
            result = subprocess.run(
                ["git", "log", "--format=%s", "-1", sha],
                cwd=ws, capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""
