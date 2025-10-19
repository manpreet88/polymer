"""Lightweight retrieval-augmented knowledge base for polymer records."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import KnowledgeBaseConfig
from .datamodels import EmbeddingResult, ProcessedPolymer

LOGGER = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    psmiles: str
    source: str
    metadata: Dict
    embedding: np.ndarray

    def to_serializable(self) -> Dict:
        data = {
            "psmiles": self.psmiles,
            "source": self.source,
            "metadata": self.metadata,
        }
        return data


class PolymerKnowledgeBase:
    """Stores polymer embeddings together with metadata for retrieval."""

    def __init__(self, config: Optional[KnowledgeBaseConfig] = None) -> None:
        self.config = (config or KnowledgeBaseConfig()).resolve()
        self.entries: List[KnowledgeEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        storage = self.config.storage_path
        meta_path = storage / self.config.metadata_filename
        emb_path = storage / self.config.embeddings_filename
        if not meta_path.exists() or not emb_path.exists():
            return
        try:
            metadata: List[Dict] = []
            with meta_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    metadata.append(json.loads(line))
            embeddings = np.load(emb_path)
            if embeddings.shape[0] != len(metadata):
                LOGGER.warning("Embedding count mismatch; skipping load from %s", storage)
                return
            self.entries = [
                KnowledgeEntry(
                    psmiles=item.get("psmiles", ""),
                    source=item.get("source", "unknown"),
                    metadata=item.get("metadata", {}),
                    embedding=embeddings[idx],
                )
                for idx, item in enumerate(metadata)
            ]
            LOGGER.info("Loaded %d knowledge base entries", len(self.entries))
        except Exception as exc:
            LOGGER.warning("Failed to load knowledge base from %s: %s", storage, exc)

    def save(self) -> None:
        if not self.entries:
            return
        storage = self.config.storage_path
        storage.mkdir(parents=True, exist_ok=True)
        meta_path = storage / self.config.metadata_filename
        emb_path = storage / self.config.embeddings_filename
        with meta_path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.to_serializable()) + "\n")
        embeddings = np.stack([entry.embedding for entry in self.entries], axis=0)
        np.save(emb_path, embeddings)
        LOGGER.info("Persisted %d knowledge base entries", len(self.entries))

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def add_entries(self, polymers: Sequence[ProcessedPolymer], embeddings: Sequence[EmbeddingResult]) -> None:
        if len(polymers) != len(embeddings):
            raise ValueError("Polymers and embeddings length mismatch")
        for polymer, embedding in zip(polymers, embeddings):
            self.entries.append(
                KnowledgeEntry(
                    psmiles=polymer.psmiles,
                    source=polymer.source,
                    metadata=polymer.metadata,
                    embedding=np.asarray(embedding.vector, dtype=np.float32),
                )
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def contains_psmiles(self, psmiles: str) -> bool:
        return any(entry.psmiles == psmiles for entry in self.entries)

    def _matrix(self) -> np.ndarray:
        if self.is_empty():
            return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
        return np.stack([entry.embedding for entry in self.entries], axis=0)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[KnowledgeEntry, float]]:
        if self.is_empty():
            return []
        matrix = self._matrix()
        query = np.asarray(query_vector, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) + 1e-8)
        sims = (matrix @ query) / (norms + 1e-8)
        top_indices = np.argsort(-sims)[:top_k]
        return [(self.entries[idx], float(sims[idx])) for idx in top_indices]

    def filter_by_metadata(self, key: str, value) -> List[KnowledgeEntry]:
        return [entry for entry in self.entries if entry.metadata.get(key) == value]

    def describe(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for entry in self.entries:
            summary[entry.source] = summary.get(entry.source, 0) + 1
        return summary


__all__ = ["PolymerKnowledgeBase", "KnowledgeEntry"]
