"""Top-level package for the advanced polymer AI agent pipeline.

This module exposes the primary orchestration classes and utilities so
consumers can construct autonomous multimodal polymer design workflows
without needing to understand the entire directory layout.  The package
is deliberately lightweight; heavier dependencies are imported lazily by
submodules when first used so the agent can run in constrained
environments where certain scientific libraries may not be installed.
"""

from .config import AgentConfig, KnowledgeBaseConfig, ModelRegistryPaths
from .data import MultimodalDataBundle, MultimodalDataExtractor
from .knowledge_base import PolymerKnowledgeBase
from .retrieval import HybridRetriever, RetrievalResult
from .orchestrator import PolymerAgentOrchestrator, AgentMode

__all__ = [
    "AgentConfig",
    "KnowledgeBaseConfig",
    "ModelRegistryPaths",
    "MultimodalDataBundle",
    "MultimodalDataExtractor",
    "PolymerKnowledgeBase",
    "HybridRetriever",
    "RetrievalResult",
    "PolymerAgentOrchestrator",
    "AgentMode",
]
