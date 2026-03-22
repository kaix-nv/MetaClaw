"""Test a kernel implementation against TileGym test suite using Docker.

Replaces a kernel file in TileGym, runs pytest inside the compute-eval
Docker container, and reports pass/fail. Restores the original after testing.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "local/compute-eval-python:13.1.0"


class KernelTester:
    """Test kernel implementations by running TileGym pytest in Docker."""

    def __init__(self, tilegym_dir: str, docker_image: str = _DEFAULT_IMAGE):
        self._tilegym_dir = Path(tilegym_dir).resolve()
        self._kernel_dir = self._tilegym_dir / "src" / "tilegym" / "ops" / "cutile"
        self._test_dir = self._tilegym_dir / "tests" / "ops"
        self._docker_image = docker_image

    def test_kernel(self, kernel_name: str, solution_code: str, timeout: int = 120) -> dict:
        """Replace kernel, run pytest in Docker, restore original.

        Returns {passed, output, kernel_name}.
        """
        kernel_file = self._kernel_dir / f"{kernel_name}.py"
        test_file = self._test_dir / f"test_{kernel_name}.py"
        backup_file = self._kernel_dir / f"{kernel_name}.py.bak"

        if not kernel_file.exists():
            return {"passed": False, "output": f"Kernel file not found: {kernel_file}", "kernel_name": kernel_name}
        if not test_file.exists():
            return {"passed": False, "output": f"Test file not found: {test_file}", "kernel_name": kernel_name}

        # Backup original
        shutil.copy2(kernel_file, backup_file)

        try:
            # Write agent's solution
            kernel_file.write_text(solution_code, encoding="utf-8")

            # Run pytest inside Docker container
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    "-v", f"{self._tilegym_dir}:/workspace/tilegym:rw",
                    "-w", "/workspace/tilegym",
                    "-e", "CUDA_TILE_CACHE_DIR=/tmp/cutile-cache",
                    self._docker_image,
                    "bash", "-c",
                    f"pip install -e . -q 2>/dev/null && "
                    f"python -m pytest tests/ops/test_{kernel_name}.py -x --quick-run -v 2>&1"
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 60,  # extra buffer for Docker startup
            )

            passed = result.returncode == 0
            output = result.stdout + result.stderr

            logger.info(
                "[KernelTester] %s: %s (exit=%d)",
                kernel_name, "PASSED" if passed else "FAILED", result.returncode,
            )

            return {
                "passed": passed,
                "output": output[-2000:] if len(output) > 2000 else output,
                "kernel_name": kernel_name,
            }

        except subprocess.TimeoutExpired:
            logger.warning("[KernelTester] %s: TIMEOUT after %ds", kernel_name, timeout)
            return {
                "passed": False,
                "output": f"Timeout after {timeout}s",
                "kernel_name": kernel_name,
            }
        except Exception as e:
            logger.error("[KernelTester] %s: ERROR %s", kernel_name, e)
            return {
                "passed": False,
                "output": str(e),
                "kernel_name": kernel_name,
            }
        finally:
            # Always restore original
            if backup_file.exists():
                shutil.move(str(backup_file), str(kernel_file))
