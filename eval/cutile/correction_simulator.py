"""Generate corrections by comparing agent solutions against TileGym references."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Key cuTile API patterns to look for
_CT_API_PATTERNS = [
    r"ct\.kernel", r"ct\.launch", r"ct\.load", r"ct\.store",
    r"ct\.gather", r"ct\.scatter", r"ct\.mma", r"ct\.matmul",
    r"ct\.bid", r"ct\.arange", r"ct\.sum", r"ct\.max",
    r"ct\.exp", r"ct\.full", r"ct\.num_tiles", r"ct\.num_blocks",
    r"ct\.Constant", r"ct\.cdiv", r"ct\.where", r"ct\.softmax",
    r"ct\.reshape", r"ct\.transpose", r"ct\.broadcast_to",
    r"ct\.atomic_add", r"ct\.expand_dims",
]

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

If the student's code is essentially correct (uses the same key APIs and
patterns, even if variable names or formatting differ), respond with just "CORRECT".
"""

_FALLBACK_PREFIX = "No, that's wrong. "


def _extract_api_calls(code: str) -> set[str]:
    """Extract ct.* API calls from code."""
    return {m.group() for p in _CT_API_PATTERNS for m in re.finditer(p, code)}


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
        """Compare agent solution to reference.

        Uses two-tier comparison:
        1. API pattern match: if agent uses all key APIs from reference → CORRECT
        2. LLM judge: for borderline cases
        """
        # Tier 1: Check if agent uses the same key ct.* APIs as reference
        ref_apis = _extract_api_calls(reference)
        agent_apis = _extract_api_calls(agent_solution)

        if ref_apis and ref_apis.issubset(agent_apis):
            # Agent uses all the key APIs from reference — likely correct
            logger.info("[CorrectionSim] %s: API match (ref=%s, agent=%s)",
                       kernel_name, ref_apis, agent_apis)
            return {"is_correct": True, "correction": "", "kernel_name": kernel_name}

        # Tier 2: LLM comparison (with larger context for big kernels)
        max_chars = min(4000, max(2000, len(reference)))
        prompt = _COMPARE_PROMPT.format(
            agent_solution=agent_solution[:max_chars],
            reference=reference[:max_chars],
        )
        response = self._llm.complete(prompt, max_tokens=500)
        response = response.strip()

        if response.upper().startswith("CORRECT"):
            return {"is_correct": True, "correction": "", "kernel_name": kernel_name}

        # Log missing APIs for debugging
        missing = ref_apis - agent_apis
        if missing:
            logger.info("[CorrectionSim] %s: missing APIs %s", kernel_name, missing)

        correction = response
        if not any(
            correction.lower().startswith(prefix)
            for prefix in ["no,", "no ", "that's wrong", "that is wrong"]
        ):
            correction = _FALLBACK_PREFIX + correction

        return {"is_correct": False, "correction": correction, "kernel_name": kernel_name}
