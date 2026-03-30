"""Main entry point for iterative solver + skill synthesis."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.cutile.iterative_solver import IterativeSolver
from eval.cutile.trace_parser import TraceParser
from eval.cutile.skill_synthesizer import SkillSynthesizer
from eval.cutile.llm_client import LLMClient
from eval.cutile.skill_writer import SkillWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Iterative cuTile Solver + Skill Synthesis")
    parser.add_argument("--tilegym-dir", required=True, help="Path to TileGym repo")
    parser.add_argument("--results-dir", default="eval/cutile/synthesis-results", help="Output dir")
    parser.add_argument("--base-skill", default="", help="Path to base cutile-python skill dir")
    parser.add_argument("--model", default="aws/anthropic/bedrock-claude-opus-4-6")
    parser.add_argument("--max-steps", type=int, default=48, help="Max OpenCode tool calls per kernel")
    parser.add_argument("--skip-solve", action="store_true", help="Skip solving, use existing traces")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Solve ---
    if not args.skip_solve:
        logger.info("=== Step 1: Iterative Solving ===")
        skill_dir = args.base_skill or str(results_dir / "base-skill")
        solver = IterativeSolver(
            tilegym_dir=args.tilegym_dir,
            skill_dir=skill_dir,
            max_steps=args.max_steps,
            model=f"nvidia/{args.model}",
        )
        solve_results = solver.solve_all(str(results_dir / "solve"))
        solved = sum(1 for r in solve_results if r["passed"])
        logger.info("Solved: %d/%d kernels", solved, len(solve_results))
    else:
        logger.info("=== Step 1: Skipped (using existing traces) ===")
        solve_results = []
        results_jsonl = results_dir / "solve" / "results.jsonl"
        if results_jsonl.exists():
            with open(results_jsonl) as f:
                solve_results = [json.loads(l) for l in f if l.strip()]

    # --- Step 2: Parse Traces ---
    logger.info("=== Step 2: Parsing Traces ===")
    trace_parser = TraceParser()
    all_pairs = []

    for result in solve_results:
        if not result["passed"]:
            continue  # Only extract from successful solves
        trace_path = result.get("trace_path", "")
        if trace_path and os.path.exists(trace_path):
            trace_text = Path(trace_path).read_text(encoding="utf-8")
            pairs = trace_parser.parse(trace_text)
            for pair in pairs:
                pair["kernel_name"] = result["kernel_name"]
            all_pairs.extend(pairs)
            logger.info("  %s: %d error→fix pairs", result["kernel_name"], len(pairs))

    logger.info("Total error→fix pairs: %d", len(all_pairs))

    # Save pairs
    with open(results_dir / "error_fix_pairs.jsonl", "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # --- Step 3: Synthesize Skills ---
    logger.info("=== Step 3: Synthesizing Skills ===")
    llm = LLMClient(model=args.model)
    skill_dir = str(results_dir / "evolved-skills")
    synthesizer = SkillSynthesizer(llm=llm, skill_dir=skill_dir)
    synth_result = synthesizer.run(all_pairs, output_dir=str(results_dir))
    logger.info("Synthesized: %d notes → %d skills", synth_result["num_notes"], synth_result["num_skills"])

    # --- Step 4: Merge Skills ---
    if args.base_skill:
        logger.info("=== Step 4: Merging Skills ===")
        evolved_skills = list(synthesizer._skill_manager._iter_all_skills())
        merged_dir = results_dir / "cutile-python-synthesized"

        # Copy base skill dir
        import shutil
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        shutil.copytree(args.base_skill, str(merged_dir))

        # Append evolved skills to SKILL.md
        skill_md = merged_dir / "SKILL.md"
        with open(skill_md, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n# Evolved Skills (from debug journeys)\n\n")
            for skill in evolved_skills:
                f.write(f"## {skill['name']}\n_{skill['description']}_\n\n{skill['content']}\n\n")

        logger.info("Merged skill dir: %s", merged_dir)

    # --- Summary ---
    logger.info("=== Summary ===")
    logger.info("  Kernels solved: %d/%d", sum(1 for r in solve_results if r["passed"]), len(solve_results))
    logger.info("  Error→fix pairs: %d", len(all_pairs))
    logger.info("  Teaching notes: %d", synth_result["num_notes"])
    logger.info("  Skills evolved: %d", synth_result["num_skills"])
    logger.info("  Results dir: %s", results_dir)


if __name__ == "__main__":
    main()
