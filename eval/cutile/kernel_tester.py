"""Test a kernel implementation against TileGym test suite using Docker.

Uses a Docker image with TileGym pre-installed. Replaces a kernel file,
runs pytest inside the container, and reports pass/fail.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "local/compute-eval-tilegym:13.1.0"


class KernelTester:
    """Test kernel implementations by running TileGym pytest in Docker."""

    def __init__(self, tilegym_dir: str, docker_image: str = _DEFAULT_IMAGE):
        self._tilegym_dir = Path(tilegym_dir).resolve()
        self._kernel_dir = self._tilegym_dir / "src" / "tilegym" / "ops" / "cutile"
        self._test_dir = self._tilegym_dir / "tests" / "ops"
        self._docker_image = docker_image

    def test_kernel(self, kernel_name: str, solution_code: str, timeout: int = 180) -> dict:
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

            # Run pytest inside Docker
            # Mount only the kernel file (not whole TileGym — it's pre-installed in image)
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    # Mount the modified kernel file over the installed one
                    "-v", f"{kernel_file}:/opt/tilegym/src/tilegym/ops/cutile/{kernel_name}.py:ro",
                    "-e", "CUDA_TILE_CACHE_DIR=/tmp/cutile-cache",
                    self._docker_image,
                    "python", "-m", "pytest",
                    f"/opt/tilegym/tests/ops/test_{kernel_name}.py",
                    "-x", "--quick-run", "-v", "-p", "no:cacheprovider",
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 60,
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
