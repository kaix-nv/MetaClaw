"""Generate corrections by comparing agent solutions against TileGym references."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_COMPARE_PROMPT = """\
Compare this student's cuTile kernel with the reference implementation.

Student attempt:
```python
{agent_solution}
```

Reference (correct):
```python
{reference}
```

If the student's code has errors or uses wrong patterns, write a correction
(2-3 sentences). Start with "No, that's wrong." Be specific about cuTile
API usage (ct.* functions, tile sizes, TMA, tensor cores).

IMPORTANT: Do NOT use "always", "never", "from now on", "remember this",
or "save this as a skill". Only describe what is wrong and what to do instead.

If the student's code is essentially correct, respond with just "CORRECT".
"""

_FALLBACK_PREFIX = "No, that's wrong. "


class CorrectionSimulator:
    """Generates natural corrections from reference comparison."""

    def __init__(self, llm):
        self._llm = llm

    def compare(
        self,
        agent_solution: str,
        reference: str,
        kernel_name: str,
    ) -> dict:
        """Compare agent solution to reference. Returns {is_correct, correction, kernel_name}."""
        prompt = _COMPARE_PROMPT.format(
            agent_solution=agent_solution[:1500],
            reference=reference[:1500],
        )
        response = self._llm.complete(prompt, max_tokens=500)
        response = response.strip()

        if response.upper().startswith("CORRECT"):
            return {"is_correct": True, "correction": "", "kernel_name": kernel_name}

        # Ensure correction triggers implicit detection
        correction = response
        if not any(
            correction.lower().startswith(prefix)
            for prefix in ["no,", "no ", "that's wrong", "that is wrong"]
        ):
            correction = _FALLBACK_PREFIX + correction

        return {"is_correct": False, "correction": correction, "kernel_name": kernel_name}
