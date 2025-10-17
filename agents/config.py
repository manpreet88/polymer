"""Configuration dataclasses for polymer agent system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for calling an LLM model."""

    model: str = "gpt-4o"
    temperature: float = 0.2
    max_output_tokens: int = 1200
    reasoning: Optional[Dict[str, str]] = field(
        default_factory=lambda: {"effort": "medium"}
    )


@dataclass
class AgentConfig:
    """Metadata for constructing a specialist agent."""

    name: str
    system_prompt: str
    model_config: ModelConfig = field(default_factory=ModelConfig)
