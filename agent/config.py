"""Configuration objects for the Polymer RAG agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class CheckpointConfig:
    """Locations of pretrained weights used by the agent.

    The defaults mirror the directories created by the training scripts in this
    repository. Paths are resolved lazily, allowing the agent to operate even
    when some checkpoints are absent. When a path exists, the agent will attempt
    to load the weights and expose the modality specific encoder.
    """

    gine_dir: Path = Path("gin_output/best")
    schnet_dir: Path = Path("schnet_output/best")
    fingerprint_dir: Path = Path("fingerprint_mlm_output/best")
    psmlm_dir: Path = Path("polybert_output/best")
    multimodal_dir: Path = Path("multimodal_output/best")

    def existing(self) -> Dict[str, Path]:
        return {
            name: path
            for name, path in {
                "gine": self.gine_dir,
                "schnet": self.schnet_dir,
                "fingerprint": self.fingerprint_dir,
                "psmiles": self.psmlm_dir,
                "multimodal": self.multimodal_dir,
            }.items()
            if path.exists()
        }


@dataclass
class RetrievalConfig:
    """Settings that control similarity search inside the knowledge base."""

    embedding_key_priority: Iterable[str] = field(
        default_factory=lambda: ("multimodal", "psmiles", "fingerprint", "gine", "schnet")
    )
    top_k: int = 5
    min_similarity: float = 0.1


@dataclass
class LLMConfig:
    """Configuration for GPT-4 orchestration via the OpenAI API."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 800
    system_prompt: str = (
        "You are PolyRAG, an assistant that reasons about polymer chemistry. "
        "Ground every answer in the retrieved evidence. Provide concise "
        "citations and outline follow-up experiment ideas when appropriate."
    )
    user_prefix: str = "User"
    assistant_prefix: str = "PolyRAG"


@dataclass
class AgentConfig:
    """Top-level configuration bundle for :class:`PolymerRAGAgent`."""

    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache_dir: Optional[Path] = Path("agent_cache")

    def ensure_cache(self) -> Path:
        path = Path(self.cache_dir) if self.cache_dir is not None else Path("agent_cache")
        path.mkdir(parents=True, exist_ok=True)
        return path


__all__ = [
    "AgentConfig",
    "CheckpointConfig",
    "LLMConfig",
    "RetrievalConfig",
]
