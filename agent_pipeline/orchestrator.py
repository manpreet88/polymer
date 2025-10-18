"""GPT-4 orchestrator that can call registered tools to answer polymer design queries."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from .config import OpenAIConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    function: Callable[[Dict[str, Any]], Any]
    parameters: Optional[Dict[str, Any]] = None

    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                },
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable[[Dict[str, Any]], Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tools[name] = ToolSpec(name=name, description=description, function=func, parameters=parameters)
        LOGGER.info("Registered tool %s", name)

    def as_openai(self) -> List[Dict[str, Any]]:
        return [tool.to_openai() for tool in self._tools.values()]

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool {name} is not registered")
        return self._tools[name].function(arguments)


class GPT4Orchestrator:
    """Thin wrapper around the OpenAI API for multi-tool conversations."""

    def __init__(
        self,
        config: Optional[OpenAIConfig] = None,
        *,
        system_prompt: Optional[str] = None,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.config = (config or OpenAIConfig.from_env())
        self.config.require_api_key()
        self.client = OpenAI(api_key=self.config.api_key, organization=self.config.organization)
        self.registry = registry or ToolRegistry()
        self.system_prompt = system_prompt or (
            "You are an expert assistant helping users explore polymer designs. "
            "Use the available tools when they provide better answers."
        )

    def chat(self, user_prompt: str, *, max_turns: int = 6) -> str:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for _ in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=self.registry.as_openai() or None,
                tool_choice="auto",
                temperature=0.2,
            )
            message = response.choices[0].message
            if message.tool_calls:
                for call in message.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments or "{}")
                    LOGGER.info("Orchestrator invoking %s with %s", name, args)
                    try:
                        result = self.registry.call(name, args)
                    except Exception as exc:
                        result = {"error": str(exc)}
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call],
                    })
                    messages.append({
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result, default=str),
                    })
                continue
            if message.content:
                return message.content
            messages.append({"role": "assistant", "content": message.content or ""})
        raise RuntimeError("Max turns reached without final response")


__all__ = ["GPT4Orchestrator", "ToolRegistry", "ToolSpec"]
