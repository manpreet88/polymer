"""Agent framework for multimodal polymer AI workflows."""

from .config import ModelConfig, AgentConfig
from .clients import OpenAIChatClient
from .base import AgentContext, BaseLLMAgent
from .specialists import (
    LiteratureResearchAgent,
    DataCurationAgent,
    PropertyPredictionAgent,
    PolymerGenerationAgent,
    VisualizationAgent,
)
from .visualization import PolymerVisualizationToolkit, VisualizationResult
from .orchestrator import GPT4Orchestrator, build_default_orchestrator

__all__ = [
    "ModelConfig",
    "AgentConfig",
    "OpenAIChatClient",
    "AgentContext",
    "BaseLLMAgent",
    "LiteratureResearchAgent",
    "DataCurationAgent",
    "PropertyPredictionAgent",
    "PolymerGenerationAgent",
    "VisualizationAgent",
    "GPT4Orchestrator",
    "build_default_orchestrator",
    "PolymerVisualizationToolkit",
    "VisualizationResult",
]
