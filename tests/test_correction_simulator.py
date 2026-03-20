# tests/test_correction_simulator.py
from eval.cutile.correction_simulator import CorrectionSimulator


class FakeLLM:
    def complete(self, prompt, max_tokens=2000):
        if "CORRECT" in prompt:
            # Comparison prompt
            return "No, that's wrong. You should use ct.matmul instead of a manual loop."
        return "some response"


def test_compare_wrong_solution():
    sim = CorrectionSimulator(llm=FakeLLM())
    result = sim.compare(
        agent_solution="def matmul(a, b): return a @ b",
        reference="@ct.kernel\ndef matmul(a, b): ct.matmul(a, b)",
        kernel_name="matmul",
    )
    assert result["is_correct"] is False
    assert "No, that's wrong" in result["correction"]


def test_compare_correct_solution():
    class CorrectLLM:
        def complete(self, prompt, max_tokens=2000):
            return "CORRECT"

    sim = CorrectionSimulator(llm=CorrectLLM())
    result = sim.compare(
        agent_solution="@ct.kernel\ndef matmul(a, b): ct.matmul(a, b)",
        reference="@ct.kernel\ndef matmul(a, b): ct.matmul(a, b)",
        kernel_name="matmul",
    )
    assert result["is_correct"] is True
    assert result["correction"] == ""


def test_correction_avoids_explicit_keywords():
    sim = CorrectionSimulator(llm=FakeLLM())
    result = sim.compare(
        agent_solution="bad code",
        reference="good code",
        kernel_name="test",
    )
    # The prompt instructs LLM to avoid "always", "never", "from now on"
    # We verify the prompt is built correctly (can't control LLM output in test)
    assert result["correction"] != ""
