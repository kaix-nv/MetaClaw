"""Main orchestrator for cuTile skill evolution evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add repo root to path so eval package is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.cutile.tilegym_loader import TileGymLoader
from eval.cutile.llm_client import LLMClient
from eval.cutile.phase_a_study import PhaseAStudy
from eval.cutile.phase_b_practice import PhaseBPractice
from eval.cutile.phase_c_evaluate import PhaseCEvaluate
from eval.cutile.skill_writer import SkillWriter
from eval.cutile.report import parse_eval_results, generate_comparison, generate_summary_text, save_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="cuTile Skill Evolution Evaluation")
    parser.add_argument("--tilegym-dir", required=True, help="Path to TileGym repo")
    parser.add_argument("--eval-kit-dir", default="", help="Path to cutile-eval-kit")
    parser.add_argument("--results-dir", default="eval/cutile/results", help="Output directory")
    parser.add_argument("--base-skill", default="", help="Path to cutile-minimal-skill.md")
    parser.add_argument("--model", default="azure/openai/gpt-5.2", help="LLM model")
    parser.add_argument("--evolver-model", default="", help="Model for skill evolution")
    parser.add_argument("--rounds", type=int, default=2, help="Max Phase B rounds")
    parser.add_argument("--task-ids", default="", help="Comma-separated task IDs (empty=all)")
    parser.add_argument("--method", default="all", choices=["all", "direct", "multiturn", "evaluate"],
                        help="Which method to run")
    parser.add_argument("--skills-from", default="", help="Pre-evolved skills dir (for --method evaluate)")
    parser.add_argument("--local-test", action="store_true", help="Skip Phase C (no GPU needed)")
    parser.add_argument("--mock-eval-results", default="", help="Mock eval results for local test")
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    llm = LLMClient(model=args.evolver_model or args.model)
    loader = TileGymLoader(args.tilegym_dir)
    pairs = loader.get_all_pairs()
    logger.info("Loaded %d kernel pairs from TileGym", len(pairs))

    # --- Method 1: Direct Extraction ---
    if args.method in ("all", "direct"):
        logger.info("=== Method 1: Direct Extraction ===")
        m1_skill_dir = os.path.join(results_dir, "method-1-direct", "evolved-skills")
        study = PhaseAStudy(llm=llm, skill_dir=m1_skill_dir)
        m1_result = study.run(
            pairs,
            output_path=os.path.join(results_dir, "method-1-direct", "teaching-summaries.jsonl"),
        )
        logger.info("Method 1 result: %s", m1_result)

        # Write merged skill file
        if args.base_skill:
            m1_skills = list(study._skill_manager._iter_all_skills())
            SkillWriter.merge(
                args.base_skill, m1_skills,
                os.path.join(results_dir, "method-1-direct", "cutile-skill-merged.md"),
            )

    # --- Method 2: Multi-Turn Evolution ---
    if args.method in ("all", "multiturn"):
        logger.info("=== Method 2: Multi-Turn Evolution ===")
        # Phase A first (same as Method 1)
        m2_skill_dir = os.path.join(results_dir, "method-2-multiturn", "evolved-skills")
        study_m2 = PhaseAStudy(llm=llm, skill_dir=m2_skill_dir)
        study_m2.run(
            pairs,
            output_path=os.path.join(results_dir, "method-2-multiturn", "phase-a", "teaching-summaries.jsonl"),
        )

        # Phase B
        practice = PhaseBPractice(
            llm=llm,
            skill_dir=m2_skill_dir,
            max_rounds=args.rounds,
        )
        m2_result = practice.run(
            pairs,
            output_dir=os.path.join(results_dir, "method-2-multiturn", "phase-b"),
        )
        logger.info("Method 2 result: %s", m2_result)

        # Write merged skill file
        if args.base_skill:
            m2_skills = list(practice._skill_manager._iter_all_skills())
            SkillWriter.merge(
                args.base_skill, m2_skills,
                os.path.join(results_dir, "method-2-multiturn", "cutile-skill-merged.md"),
            )

    # --- Phase C: Evaluate on compute-eval ---
    # Skip Phase C entirely in local-test mode (no GPU required)
    if args.method in ("all", "evaluate") and not args.local_test:
        if not args.eval_kit_dir:
            logger.error("--eval-kit-dir required for Phase C")
            return

        logger.info("=== Phase C: Compute-Eval Benchmark ===")
        evaluator = PhaseCEvaluate(args.eval_kit_dir, model=args.model)

        # Run for each condition and collect results
        # (actual B200 runs — this takes hours)
        logger.info("Phase C requires B200 allocation. Run each condition separately.")
        logger.info("Use: --method evaluate --skills-from <path>")

    # --- Comparison report (if mock results available) ---
    if args.mock_eval_results and args.local_test:
        logger.info("=== Generating comparison from mock results ===")
        # For local testing, use the same mock for all three conditions
        mock = parse_eval_results(args.mock_eval_results)
        report = generate_comparison(mock, mock, mock)
        summary = generate_summary_text(report)
        save_report(report, summary, results_dir)
        print(summary)


if __name__ == "__main__":
    main()
