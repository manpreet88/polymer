"""Base agent abstractions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

from .clients import LLMMessage, OpenAIChatClient
from .config import AgentConfig, ModelConfig


@dataclass
class AgentContext:
    """Shared context that is passed between orchestrator and specialists."""

    dataset_description: str = ""
    project_goals: str = ""
    constraints: Dict[str, str] = field(default_factory=dict)
    additional_notes: Dict[str, str] = field(default_factory=dict)

    def to_bullet_string(self) -> str:
        bullets: List[str] = []
        if self.dataset_description:
            bullets.append(f"Dataset: {self.dataset_description}")
        if self.project_goals:
            bullets.append(f"Goals: {self.project_goals}")
        if self.constraints:
            constraints = ", ".join(f"{k}: {v}" for k, v in self.constraints.items())
            bullets.append(f"Constraints: {constraints}")
        if self.additional_notes:
            notes = ", ".join(f"{k}: {v}" for k, v in self.additional_notes.items())
            bullets.append(f"Notes: {notes}")
        return "\n".join(f"- {line}" for line in bullets)


class BaseLLMAgent:
    """Base class for all LLM-powered agents."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        client: OpenAIChatClient,
    ) -> None:
        self.config = config
        self.client = client

    @property
    def name(self) -> str:
        return self.config.name

    def build_messages(
        self,
        *,
        task: str,
        context: Optional[AgentContext] = None,
        history: Optional[Iterable[Mapping[str, str]]] = None,
    ) -> List[LLMMessage]:
        ctx_str = context.to_bullet_string() if context else ""
        user_content = task
        if ctx_str:
            user_content = f"Context:\n{ctx_str}\n\nTask:\n{task}"
        messages: List[LLMMessage] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]
        if history:
            messages.extend(history)
        return messages

    def run(
        self,
        *,
        task: str,
        context: Optional[AgentContext] = None,
        history: Optional[Iterable[Mapping[str, str]]] = None,
        model_override: Optional[ModelConfig] = None,
        json_mode: bool = False,
    ) -> str:
        messages = self.build_messages(task=task, context=context, history=history)
        config = model_override or self.config.model_config
        return self.client.complete(messages, model_config=config, json_mode=json_mode)

    def run_json(
        self,
        *,
        task: str,
        context: Optional[AgentContext] = None,
        history: Optional[Iterable[Mapping[str, str]]] = None,
        model_override: Optional[ModelConfig] = None,
    ) -> Dict[str, object]:
        response = self.run(
            task=task,
            context=context,
            history=history,
            model_override=model_override,
            json_mode=True,
        )
        return json.loads(response)
