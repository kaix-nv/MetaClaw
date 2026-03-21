"""Thin LLM API client for the eval harness."""

from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        from openai import OpenAI

        self._model = model or os.environ.get(
            "SKILL_EVOLVER_MODEL", "aws/anthropic/bedrock-claude-opus-4-6"
        )
        api_key = api_key or os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://inference-api.nvidia.com/v1"
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        """Single-turn completion."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def multi_turn(self, messages: list[dict], max_tokens: int = 2000) -> str:
        """Multi-turn conversation. messages: [{"role": "user"/"assistant", "content": "..."}]"""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content
