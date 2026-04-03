"""Autoresearch-style cuTile solver: program.md-driven OpenCode in Docker."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from .tilegym_loader import TileGymLoader
from .git_trace_parser import GitTraceParser

logger = logging.getLogger(__name__)

_DEFAULT_OPENCODE_IMAGE = "local/opencode-agent:cutile-synthv2"
_PROGRAM_MD = Path(__file__).parent / "program.md"


class AutoresearchSolver:
    """Solve TileGym kernels using autoresearch pattern with program.md."""

    def __init__(
        self,
        tilegym_dir: str,
        skill_dir: str,
        opencode_image: str = _DEFAULT_OPENCODE_IMAGE,
        timeout: int = 600,
        model: str = "nvidia/aws/anthropic/bedrock-claude-opus-4-6",
    ):
        self._loader = TileGymLoader(tilegym_dir)
        self._tilegym_dir = Path(tilegym_dir).resolve()
        self._skill_dir = Path(skill_dir).resolve()
        self._opencode_image = opencode_image
        self._timeout = timeout
        self._model = model
        self._trace_parser = GitTraceParser()

    def _create_workspace(self, kernel_name: str, workspace_dir: Path) -> None:
        """Create a solve workspace with program.md, problem spec, test, and skill."""
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True)

        # 1. Copy program.md
        shutil.copy2(_PROGRAM_MD, workspace_dir / "program.md")

        # 2. Write problem spec
        spec = {
            "kernel": kernel_name,
            "description": f"Implement the {kernel_name} cuTile kernel that passes the test in test/",
        }
        (workspace_dir / "problem-spec.yaml").write_text(
            json.dumps(spec, indent=2), encoding="utf-8"
        )

        # 3. Copy test file (VISIBLE to agent)
        test_dir = workspace_dir / "test"
        test_dir.mkdir()
        test_source = self._loader.get_test_source(kernel_name)
        if test_source:
            (test_dir / f"test_{kernel_name}.py").write_text(test_source, encoding="utf-8")

        # Copy test utilities
        for fname in ["common.py", "conftest.py", "__init__.py"]:
            src = self._tilegym_dir / "tests" / fname
            if src.exists():
                (test_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        # Also create tests/__init__.py for import resolution
        tests_pkg = test_dir.parent / "tests"
        if not tests_pkg.exists():
            tests_pkg.mkdir(exist_ok=True)
            (tests_pkg / "__init__.py").write_text("", encoding="utf-8")

        # 4. Copy skill dir into workspace
        skill_dest = workspace_dir / ".opencode" / "skill" / "cutile-python"
        shutil.copytree(self._skill_dir, skill_dest)

        # 5. Create empty results.log
        (workspace_dir / "results.log").write_text("", encoding="utf-8")

    def solve_kernel(self, kernel_name: str, output_dir: Path) -> dict:
        """Solve a single kernel. Returns result dict."""
        workspace = output_dir / "workspaces" / kernel_name
        self._create_workspace(kernel_name, workspace)

        # Build OpenCode config
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

        # Launch OpenCode in Docker
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    "--entrypoint", "",
                    "-v", f"{workspace}:/testbed:rw",
                    "-e", f"OPENCODE_CONFIG_CONTENT={opencode_config}",
                    "-e", f"OPENAI_API_KEY={os.environ.get('API_KEY', os.environ.get('OPENAI_API_KEY', ''))}",
                    "-e", "CUDA_TILE_CACHE_DIR=/tmp/cutile-cache",
                    "-e", "XDG_DATA_HOME=/tmp/xdg-data",
                    "-e", "XDG_CONFIG_HOME=/tmp/xdg-config",
                    "-e", "XDG_STATE_HOME=/tmp/xdg-state",
                    "-w", "/testbed",
                    self._opencode_image,
                    "opencode", "run",
                    "-m", self._model,
                    "--dir", "/testbed",
                    "Follow the instructions in program.md",
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired:
            logger.warning("[AutoSolver] %s: TIMEOUT after %ds", kernel_name, self._timeout)
            stdout = ""
            stderr = "TIMEOUT"
        except Exception as e:
            logger.error("[AutoSolver] %s: ERROR %s", kernel_name, e)
            stdout = ""
            stderr = str(e)

        # Parse results
        status = self._trace_parser.get_final_status(str(workspace))
        pairs = self._trace_parser.parse(str(workspace))

        logger.info(
            "[AutoSolver] %s: %s (attempts=%d, pairs=%d)",
            kernel_name,
            "SOLVED" if status["passed"] else "UNSOLVED",
            status["attempts"],
            len(pairs),
        )

        return {
            "kernel_name": kernel_name,
            "passed": status["passed"],
            "attempts": status["attempts"],
            "num_pairs": len(pairs),
            "pairs": pairs,
            "workspace": str(workspace),
            "stderr_tail": stderr[-500:] if stderr else "",
        }

    def solve_all(self, output_dir: str) -> list[dict]:
        """Solve all TileGym kernels. Returns list of results."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        kernels = self._loader.list_kernels()
        results = []

        for i, kernel_name in enumerate(kernels):
            logger.info("[AutoSolver] Solving %s (%d/%d)", kernel_name, i + 1, len(kernels))
            result = self.solve_kernel(kernel_name, output)
            results.append(result)

            # Save incrementally
            with open(output / "results.jsonl", "a") as f:
                # Don't write pairs to results file (too large)
                r = {k: v for k, v in result.items() if k != "pairs"}
                f.write(json.dumps(r) + "\n")

        solved = sum(1 for r in results if r["passed"])
        logger.info("[AutoSolver] Done: %d/%d solved", solved, len(results))
        return results
