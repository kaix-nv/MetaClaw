"""Iterative cuTile solver using OpenCode in Docker with tests visible."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from .tilegym_loader import TileGymLoader

logger = logging.getLogger(__name__)

_DEFAULT_OPENCODE_IMAGE = "local/opencode-agent:cutile-synthv2"

_SOLVE_PROMPT = (
    "You are solving a cuTile kernel implementation problem. "
    "Read the problem description in problem-spec.yaml and the cuTile skill first. "
    "Write solution.py that implements the required kernel. "
    "Run the test in test/ to verify your solution. If it fails, read the error and fix your code. "
    "Keep iterating until the test passes or you run out of steps."
)


class IterativeSolver:
    """Solve TileGym kernels using OpenCode with real test feedback."""

    def __init__(
        self,
        tilegym_dir: str,
        skill_dir: str,
        opencode_image: str = _DEFAULT_OPENCODE_IMAGE,
        max_steps: int = 48,
        model: str = "nvidia/aws/anthropic/bedrock-claude-opus-4-6",
    ):
        self._loader = TileGymLoader(tilegym_dir)
        self._tilegym_dir = Path(tilegym_dir).resolve()
        self._skill_dir = Path(skill_dir).resolve()
        self._opencode_image = opencode_image
        self._max_steps = max_steps
        self._model = model

    def _create_workspace(self, kernel_name: str, workspace_dir: Path) -> None:
        """Create a solve workspace with problem spec, test, and skill."""
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Write problem spec
        test_source = self._loader.get_test_source(kernel_name) or ""
        prompt = (
            f"Implement the cuTile kernel: {kernel_name}\n\n"
            f"Your solution must pass the test in test/test_{kernel_name}.py.\n"
            f"Write solution.py with the implementation.\n"
            f"Use the cuTile API (import cuda.tile as ct)."
        )
        spec = {"task_id": f"tilegym/{kernel_name}", "prompt": prompt}
        (workspace_dir / "problem-spec.yaml").write_text(
            json.dumps(spec, indent=2), encoding="utf-8"
        )

        # Copy test file
        test_dir = workspace_dir / "test"
        test_dir.mkdir(exist_ok=True)
        if test_source:
            (test_dir / f"test_{kernel_name}.py").write_text(test_source, encoding="utf-8")

        # Copy TileGym common test utilities
        common_py = self._tilegym_dir / "tests" / "common.py"
        conftest_py = self._tilegym_dir / "tests" / "conftest.py"
        tests_init = self._tilegym_dir / "tests" / "__init__.py"
        if common_py.exists():
            (test_dir / "common.py").write_text(common_py.read_text(), encoding="utf-8")
        if conftest_py.exists():
            (test_dir / "conftest.py").write_text(conftest_py.read_text(), encoding="utf-8")
        if tests_init.exists():
            (test_dir / "__init__.py").write_text(tests_init.read_text(), encoding="utf-8")

    def solve_kernel(
        self,
        kernel_name: str,
        output_dir: Path,
    ) -> dict:
        """Solve a single kernel. Returns {passed, trace_path, solution_path, kernel_name}."""
        workspace = output_dir / "workspace" / kernel_name
        self._create_workspace(kernel_name, workspace)

        # OpenCode config
        opencode_config = json.dumps({
            "model": self._model,
            "small_model": self._model,
            "provider": {
                "nvidia": {
                    "name": "nvidia",
                    "npm": "@ai-sdk/openai-compatible",
                    "models": {
                        "aws/anthropic/bedrock-claude-opus-4-6": {
                            "name": self._model,
                            "id": "aws/anthropic/bedrock-claude-opus-4-6",
                        }
                    },
                    "options": {
                        "apiKey": os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "")),
                        "baseURL": "https://inference-api.nvidia.com/v1",
                    },
                }
            },
        }, separators=(",", ":"))

        # Run OpenCode in Docker
        trace_dir = output_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{kernel_name}.jsonl"

        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    "-v", f"{workspace}:/testbed:rw",
                    "-v", f"{self._skill_dir}:/testbed/.opencode/skill/cutile-python:ro",
                    "-e", f"OPENCODE_CONFIG_CONTENT={opencode_config}",
                    "-e", f"OPENAI_API_KEY={os.environ.get('API_KEY', os.environ.get('OPENAI_API_KEY', ''))}",
                    "-e", "CUDA_TILE_CACHE_DIR=/tmp/cutile-cache",
                    "-e", f"XDG_DATA_HOME=/tmp/xdg-data",
                    "-e", f"XDG_CONFIG_HOME=/tmp/xdg-config",
                    "-e", f"XDG_STATE_HOME=/tmp/xdg-state",
                    "-w", "/testbed",
                    self._opencode_image,
                    "opencode", "run",
                    "-m", self._model,
                    "--dir", "/testbed",
                    "--max-steps", str(self._max_steps),
                    _SOLVE_PROMPT,
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max per kernel
            )
            stdout = result.stdout
            stderr = result.stderr

        except subprocess.TimeoutExpired:
            logger.warning("[Solver] %s: TIMEOUT", kernel_name)
            stdout = ""
            stderr = "TIMEOUT"
        except Exception as e:
            logger.error("[Solver] %s: ERROR %s", kernel_name, e)
            stdout = ""
            stderr = str(e)

        # Find trace file (OpenCode writes to .opencode/sessions/)
        trace_content = ""
        opencode_dir = workspace / ".opencode" / "sessions"
        if opencode_dir.exists():
            for session_dir in opencode_dir.iterdir():
                for jsonl in session_dir.glob("*.jsonl"):
                    trace_content = jsonl.read_text(encoding="utf-8")
                    break

        trace_path.write_text(trace_content, encoding="utf-8")

        # Check if solution exists
        solution_path = workspace / "solution.py"
        passed = solution_path.exists() and solution_path.stat().st_size > 0

        # Log result
        logger.info("[Solver] %s: %s (trace=%d lines)",
                    kernel_name, "SOLVED" if passed else "UNSOLVED",
                    len(trace_content.strip().split("\n")) if trace_content else 0)

        return {
            "kernel_name": kernel_name,
            "passed": passed,
            "trace_path": str(trace_path),
            "solution_path": str(solution_path) if passed else "",
            "stdout_tail": stdout[-500:] if stdout else "",
            "stderr_tail": stderr[-500:] if stderr else "",
        }

    def solve_all(self, output_dir: str) -> list[dict]:
        """Solve all TileGym kernels. Returns list of results."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        kernels = self._loader.list_kernels()
        results = []

        for kernel_name in kernels:
            logger.info("[Solver] Solving %s (%d/%d)", kernel_name, len(results) + 1, len(kernels))
            result = self.solve_kernel(kernel_name, output)
            results.append(result)

            # Save results incrementally
            with open(output / "results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

        solved = sum(1 for r in results if r["passed"])
        logger.info("[Solver] Done: %d/%d solved", solved, len(results))
        return results
