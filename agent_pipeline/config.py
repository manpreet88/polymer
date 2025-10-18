"""Configuration dataclasses for the polymer AI agent pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CheckpointConfig:
    """Paths to the pretrained checkpoints used across the pipeline."""

    base_dir: Path = field(default_factory=lambda: Path("."))
    multimodal_dir: Path = field(default_factory=lambda: Path("multimodal_output") / "best")
    gine_dir: Path = field(default_factory=lambda: Path("gin_output") / "best")
    schnet_dir: Path = field(default_factory=lambda: Path("schnet_output") / "best")
    fingerprint_dir: Path = field(default_factory=lambda: Path("fingerprint_mlm_output") / "best")
    psmiles_dir: Path = field(default_factory=lambda: Path("polybert_output") / "best")
    property_heads_dir: Path = field(default_factory=lambda: Path("property_heads"))

    def resolve(self) -> "CheckpointConfig":
        """Return a copy with absolute paths."""

        resolved = CheckpointConfig(
            base_dir=self.base_dir.resolve(),
            multimodal_dir=(self.base_dir / self.multimodal_dir).resolve(),
            gine_dir=(self.base_dir / self.gine_dir).resolve(),
            schnet_dir=(self.base_dir / self.schnet_dir).resolve(),
            fingerprint_dir=(self.base_dir / self.fingerprint_dir).resolve(),
            psmiles_dir=(self.base_dir / self.psmiles_dir).resolve(),
            property_heads_dir=(self.base_dir / self.property_heads_dir).resolve(),
        )
        return resolved


@dataclass
class KnowledgeBaseConfig:
    """Configuration for the lightweight retrieval-augmented knowledge base."""

    storage_path: Path = field(default_factory=lambda: Path("knowledge_base"))
    embedding_dim: int = 600
    metadata_filename: str = "metadata.jsonl"
    embeddings_filename: str = "embeddings.npy"

    def resolve(self) -> "KnowledgeBaseConfig":
        resolved_path = (self.storage_path).resolve()
        return KnowledgeBaseConfig(
            storage_path=resolved_path,
            embedding_dim=self.embedding_dim,
            metadata_filename=self.metadata_filename,
            embeddings_filename=self.embeddings_filename,
        )


@dataclass
class OpenAIConfig:
    """Configuration required to talk to the OpenAI API."""

    api_key: Optional[str] = None
    model: str = "gpt-4o"
    organization: Optional[str] = None
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Load configuration from environment variables."""

        api_key = os.getenv("OPENAI_API_KEY")
        organization = os.getenv("OPENAI_ORG")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        timeout_env = os.getenv("OPENAI_TIMEOUT")
        timeout = float(timeout_env) if timeout_env else 60.0
        return cls(api_key=api_key, organization=organization, model=model, timeout=timeout)

    def require_api_key(self) -> None:
        """Ensure an API key is provided, raising a descriptive error otherwise."""

        if not self.api_key:
            raise RuntimeError(
                "No OpenAI API key configured. Set the OPENAI_API_KEY environment variable "
                "or provide OpenAIConfig(api_key=...)."
            )


__all__ = [
    "CheckpointConfig",
    "KnowledgeBaseConfig",
    "OpenAIConfig",
]
