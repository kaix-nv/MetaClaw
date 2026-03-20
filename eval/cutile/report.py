"""Comparison reports and skill attribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def parse_eval_results(jsonl_path: str) -> dict[str, bool]:
    """Parse *-graded-solutions.jsonl -> {task_id: passed}."""
    results = {}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            task_id = entry.get("task_id", "")
            passed = entry.get("passed", False)
            skipped = entry.get("skipped", False)
            if not skipped:
                results[task_id] = passed
    return results


def generate_comparison(
    control: dict[str, bool],
    method1: dict[str, bool],
    method2: dict[str, bool],
) -> dict:
    """Compare three conditions. Returns structured report."""
    all_tasks = sorted(set(control) | set(method1) | set(method2))

    flips_c_to_m1 = [t for t in all_tasks if not control.get(t, False) and method1.get(t, False)]
    flips_c_to_m2 = [t for t in all_tasks if not control.get(t, False) and method2.get(t, False)]
    flips_m1_to_m2 = [t for t in all_tasks if not method1.get(t, False) and method2.get(t, False)]
    reg_m1 = [t for t in all_tasks if control.get(t, False) and not method1.get(t, False)]
    reg_m2 = [t for t in all_tasks if control.get(t, False) and not method2.get(t, False)]

    return {
        "control_pass": sum(control.values()),
        "control_total": len(control),
        "method1_pass": sum(method1.values()),
        "method1_total": len(method1),
        "method2_pass": sum(method2.values()),
        "method2_total": len(method2),
        "flips_control_to_method1": flips_c_to_m1,
        "flips_control_to_method2": flips_c_to_m2,
        "flips_method1_to_method2": flips_m1_to_m2,
        "regressions_method1": reg_m1,
        "regressions_method2": reg_m2,
    }


def generate_summary_text(report: dict) -> str:
    """Human-readable summary table."""
    lines = [
        "=== cuTile Skill Evolution Report ===",
        "",
        f"{'Method':<30} {'Pass':>6} {'Total':>6} {'Delta':>6}",
        "-" * 54,
        f"{'No skills (control)':<30} {report['control_pass']:>4}/{report['control_total']:<4} {'':>6}",
        f"{'Method 1: Direct extract':<30} {report['method1_pass']:>4}/{report['method1_total']:<4} {'+' + str(report['method1_pass'] - report['control_pass']):>6}",
        f"{'Method 2: Multi-turn':<30} {report['method2_pass']:>4}/{report['method2_total']:<4} {'+' + str(report['method2_pass'] - report['control_pass']):>6}",
        "",
        f"Flips (fail->pass) Control->Method 1: {report['flips_control_to_method1']}",
        f"Flips (fail->pass) Control->Method 2: {report['flips_control_to_method2']}",
        f"Flips (fail->pass) Method 1->Method 2: {report['flips_method1_to_method2']}",
        f"Regressions Method 1: {report['regressions_method1']}",
        f"Regressions Method 2: {report['regressions_method2']}",
    ]
    return "\n".join(lines)


def save_report(report: dict, summary: str, output_dir: str) -> None:
    """Save structured and text reports."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    (Path(output_dir) / "summary.txt").write_text(summary, encoding="utf-8")
