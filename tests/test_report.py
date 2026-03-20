# tests/test_report.py
import json
from eval.cutile.report import generate_comparison, generate_summary_text


def test_generate_comparison():
    control = {"cutile/0": True, "cutile/1": False, "cutile/2": False}
    method1 = {"cutile/0": True, "cutile/1": True, "cutile/2": False}
    method2 = {"cutile/0": True, "cutile/1": True, "cutile/2": True}

    report = generate_comparison(control, method1, method2)
    assert report["control_pass"] == 1
    assert report["method1_pass"] == 2
    assert report["method2_pass"] == 3
    assert "cutile/1" in report["flips_control_to_method1"]
    assert "cutile/2" in report["flips_method1_to_method2"]


def test_generate_summary_text():
    report = {
        "control_pass": 20, "control_total": 48,
        "method1_pass": 25, "method1_total": 48,
        "method2_pass": 30, "method2_total": 48,
        "flips_control_to_method1": ["cutile/1", "cutile/2"],
        "flips_control_to_method2": ["cutile/1", "cutile/2", "cutile/3"],
        "flips_method1_to_method2": ["cutile/3"],
        "regressions_method1": [],
        "regressions_method2": [],
    }
    text = generate_summary_text(report)
    assert "20/48" in text
    assert "25/48" in text
    assert "30/48" in text
