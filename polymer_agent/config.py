"""Configuration dataclasses for the polymer agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ModelRegistryPaths:
    """Directory pointers to pretrained checkpoints for each modality.

    The defaults follow the conventions used throughout the repository,
    but the paths can be overridden at runtime for experimentation.
    """

    base_dir: Path = Path("multimodal_output")
    gine: Path = Path("multimodal_output/best/gine")
    schnet: Path = Path("multimodal_output/best/schnet")
    fingerprint: Path = Path("multimodal_output/best/fingerprint")
    deberta: Path = Path("multimodal_output/best/deberta")
    fusion: Path = Path("multimodal_output/best/fusion_head")

    def as_dict(self) -> Dict[str, Path]:
        return {
            "gine": self.gine,
            "schnet": self.schnet,
            "fingerprint": self.fingerprint,
            "deberta": self.deberta,
            "fusion": self.fusion,
        }


@dataclass
class KnowledgeBaseConfig:
    """Configuration for the polymer knowledge graph backing the RAG layer."""

    storage_path: Optional[Path] = None
    autoreload: bool = True
    embedding_key: str = "embedding"
    max_neighbors: int = 15


@dataclass
class AgentConfig:
    """Aggregate configuration for the polymer agent."""

    model_paths: ModelRegistryPaths = field(default_factory=ModelRegistryPaths)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    default_batch_size: int = 16
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    top_p: float = 0.95
    max_history: int = 6

    def resolve_paths(self, root: Path) -> None:
        """Resolve all path-like attributes relative to a repository root."""

        for attr, path in self.model_paths.as_dict().items():
            resolved = (root / path).resolve()
            setattr(self.model_paths, attr, resolved)
        if self.model_paths.base_dir:
            self.model_paths.base_dir = (root / self.model_paths.base_dir).resolve()
        if self.knowledge_base.storage_path:
            self.knowledge_base.storage_path = (
                root / self.knowledge_base.storage_path
            ).resolve()
