"""LLM client utilities."""
from __future__ import annotations

import json
import os
from typing import Iterable, List, Mapping, MutableMapping, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The `openai` package is required for the polymer agent system. "
        "Install it with `pip install openai`."
    ) from exc

from .config import ModelConfig

LLMMessage = Mapping[str, str]


class OpenAIChatClient:
    """Light wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set in the environment or provided explicitly."
            )
        self._client = OpenAI(api_key=key, base_url=base_url, organization=organization)
        self.default_model = default_model

    def complete(
        self,
        messages: Iterable[LLMMessage],
        *,
        model_config: Optional[ModelConfig] = None,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request and return the model text."""

        config = model_config or ModelConfig(model=self.default_model)
        messages_list: List[MutableMapping[str, str]] = [dict(m) for m in messages]

        response = self._client.chat.completions.create(
            model=config.model,
            messages=messages_list,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            response_format={"type": "json_object"} if json_mode else None,
        )
        message = response.choices[0].message
        content = message.content or ""

        if json_mode:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Model response was not valid JSON: {content}") from exc
            return json.dumps(parsed)
        return content
