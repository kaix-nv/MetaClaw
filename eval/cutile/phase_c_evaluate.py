"""Phase C: Run compute-eval benchmark via subprocess."""

from __future__ import annotations

import glob
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class PhaseCEvaluate:
    """Run compute-eval benchmark using cutile-eval-kit shell scripts."""

    def __init__(self, eval_kit_dir: str, model: str = "azure/openai/gpt-5.2"):
        self._eval_kit_dir = eval_kit_dir
        self._model = model
        self._solve_script = os.path.join(
            eval_kit_dir, "scripts", "compute-eval-run-cutile-opencode-gpt52.sh"
        )
        self._eval_script = os.path.join(
            eval_kit_dir, "scripts", "run_b200_local_compute_eval_evalonly.sh"
        )

    def run_solve_and_eval(
        self,
        run_root: str,
        skill_md: str = "",
        task_ids: str = "",
    ) -> str:
        """Run full solve + eval. Returns path to graded results JSONL."""
        env = {
            **os.environ,
            "RUN_ROOT": run_root,
            "MODEL": self._model,
            "TASK_IDS": task_ids,
        }
        if skill_md:
            env["SKILL_MD"] = skill_md

        logger.info("[PhaseC] Running solve+eval: %s", self._solve_script)
        subprocess.run(
            ["bash", self._solve_script],
            env=env,
            check=True,
        )

        return self._find_graded_results(run_root)

    def run_eval_only(
        self,
        base_run: str,
        run_root: str,
        task_ids: str = "",
    ) -> str:
        """Run eval-only on existing solutions. Returns path to graded results."""
        env = {
            **os.environ,
            "BASE_RUN": base_run,
            "RUN_ROOT": run_root,
            "TASK_IDS": task_ids,
        }

        logger.info("[PhaseC] Running eval-only: %s", self._eval_script)
        subprocess.run(
            ["bash", self._eval_script],
            env=env,
            check=True,
        )

        return self._find_graded_results(run_root)

    @staticmethod
    def _find_graded_results(run_root: str) -> str:
        """Find the *-graded-solutions.jsonl file in eval-output/."""
        pattern = os.path.join(run_root, "eval-output", "*-graded-solutions.jsonl")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No graded results found matching: {pattern}")
        return matches[-1]
