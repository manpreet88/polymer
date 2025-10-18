"""In-memory hybrid knowledge base for multimodal polymer data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import RetrievalConfig
from .processing import PolymerModalities


@dataclass
class KnowledgeEntry:
    identifier: str
    modalities: PolymerModalities
    embeddings: Dict[str, np.ndarray]
    summary: Dict[str, str] = field(default_factory=dict)

    def as_context_block(self) -> str:
        lines = [f"ID: {self.identifier}"]
        lines.append(f"PSMILES: {self.modalities.canonical_psmiles}")
        if self.summary:
            for key, value in sorted(self.summary.items()):
                lines.append(f"{key}: {value}")
        if self.modalities.metadata:
            for key, value in self.modalities.metadata.items():
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


class PolymerKnowledgeBase:
    """Stores polymer entries and performs simple cosine-similarity retrieval."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self.entries: List[KnowledgeEntry] = []

    def add(self, entry: KnowledgeEntry) -> None:
        self.entries.append(entry)

    def build_entry(
        self,
        identifier: str,
        sample: PolymerModalities,
        embeddings: Dict[str, np.ndarray],
    ) -> KnowledgeEntry:
        normalized = {k: self._normalize(v) for k, v in embeddings.items() if v is not None}
        return KnowledgeEntry(identifier=identifier, modalities=sample, embeddings=normalized, summary=sample.summary())

    def search(self, query_embeddings: Dict[str, np.ndarray]) -> List[Tuple[KnowledgeEntry, float]]:
        if not self.entries:
            return []
        candidates: List[Tuple[KnowledgeEntry, float]] = []
        for entry in self.entries:
            score = self._max_similarity(query_embeddings, entry.embeddings)
            if score >= self.config.min_similarity:
                candidates.append((entry, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[: self.config.top_k]

    def _max_similarity(self, query: Dict[str, np.ndarray], entry: Dict[str, np.ndarray]) -> float:
        best = 0.0
        for key in self.config.embedding_key_priority:
            if key in query and key in entry:
                best = max(best, float(np.dot(query[key], entry[key])))
        return best

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        if vec.size == 0:
            return vec
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm


__all__ = ["PolymerKnowledgeBase", "KnowledgeEntry"]
