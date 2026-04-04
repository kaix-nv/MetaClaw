"""Main entry point for autoresearch-style skill synthesis."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.cutile.autoresearch_solver import AutoresearchSolver
from eval.cutile.skill_synthesizer import SkillSynthesizer
from eval.cutile.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Autoresearch-style cuTile Skill Synthesis")
    parser.add_argument("--tilegym-dir", required=True, help="Path to TileGym repo")
    parser.add_argument("--results-dir", default="eval/cutile/synthesis-results", help="Output dir")
    parser.add_argument("--base-skill", default="", help="Path to base cutile-python skill dir")
    parser.add_argument("--model", default="aws/anthropic/bedrock-claude-opus-4-6")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per kernel (seconds)")
    parser.add_argument("--skip-solve", action="store_true", help="Skip solving, use existing results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Solve ---
    all_pairs = []

    if not args.skip_solve:
        logger.info("=== Step 1: Autoresearch Solving ===")
        skill_dir = args.base_skill or str(results_dir / "base-skill")
        solver = AutoresearchSolver(
            tilegym_dir=args.tilegym_dir,
            skill_dir=skill_dir,
            timeout=args.timeout,
            model=f"nvinference/{args.model}",
        )
        solve_results = solver.solve_all(str(results_dir / "solve"))
        solved = sum(1 for r in solve_results if r["passed"])
        logger.info("Solved: %d/%d kernels", solved, len(solve_results))

        # Collect pairs from successful solves only
        for result in solve_results:
            if result["passed"] and result["pairs"]:
                for pair in result["pairs"]:
                    pair["kernel_name"] = result["kernel_name"]
                    all_pairs.append(pair)
    else:
        logger.info("=== Step 1: Skipped (loading existing results) ===")
        pairs_file = results_dir / "error_fix_pairs.jsonl"
        if pairs_file.exists():
            with open(pairs_file) as f:
                all_pairs = [json.loads(l) for l in f if l.strip()]

    logger.info("Total error→fix pairs: %d", len(all_pairs))

    # Save pairs
    with open(results_dir / "error_fix_pairs.jsonl", "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # --- Step 2: Synthesize Skills ---
    logger.info("=== Step 2: Synthesizing Skills ===")
    llm = LLMClient(model=args.model)
    skill_dir = str(results_dir / "evolved-skills")
    synthesizer = SkillSynthesizer(llm=llm, skill_dir=skill_dir)
    synth_result = synthesizer.run(all_pairs, output_dir=str(results_dir))
    logger.info("Synthesized: %d notes → %d skills", synth_result["num_notes"], synth_result["num_skills"])

    # --- Step 3: Merge Skills ---
    if args.base_skill:
        logger.info("=== Step 3: Merging Skills ===")
        merged_dir = results_dir / "cutile-python-synthesized"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        shutil.copytree(args.base_skill, str(merged_dir))

        # Append evolved skills to SKILL.md
        evolved_skills = list(synthesizer._skill_manager._iter_all_skills())
        skill_md = merged_dir / "SKILL.md"
        with open(skill_md, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n# Evolved Skills (from debug journeys)\n\n")
            for skill in evolved_skills:
                name = skill.get("name", "unnamed")
                desc = skill.get("description", "")
                content = skill.get("content", "")
                f.write(f"## {name}\n_{desc}_\n\n{content}\n\n")

        logger.info("Merged skill dir: %s", merged_dir)

    # --- Summary ---
    logger.info("=== Summary ===")
    if not args.skip_solve:
        logger.info("  Kernels solved: %d/%d", solved, len(solve_results))
    logger.info("  Error→fix pairs: %d", len(all_pairs))
    logger.info("  Teaching notes: %d", synth_result["num_notes"])
    logger.info("  Skills evolved: %d", synth_result["num_skills"])
    logger.info("  Results dir: %s", results_dir)


if __name__ == "__main__":
    main()
