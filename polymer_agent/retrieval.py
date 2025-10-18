"""Retrieval-augmented generation utilities for polymer agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .config import KnowledgeBaseConfig
from .knowledge_base import KnowledgeNode, PolymerKnowledgeBase


@dataclass
class RetrievalResult:
    """Container describing an item pulled from the knowledge base."""

    node: KnowledgeNode
    score: float
    rationale: str

    def to_message(self) -> str:
        """Return a concise textual representation suitable for LLM prompts."""

        attrs = ", ".join(f"{key}={value}" for key, value in self.node.attributes.items())
        return f"[{self.node.node_type}] {self.node.identifier} (score={self.score:.3f}): {attrs}"


class HybridRetriever:
    """Combines embedding similarity and attribute filters for RAG."""

    def __init__(self, knowledge_base: PolymerKnowledgeBase, config: KnowledgeBaseConfig) -> None:
        self._kb = knowledge_base
        self._config = config

    def by_embedding(
        self, embedding: Sequence[float], top_k: int = 5, metric: str = "cosine"
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for node, score in self._kb.query_similar_embeddings(list(embedding), top_k=top_k, metric=metric):
            rationale = "semantic similarity in multimodal latent space"
            results.append(RetrievalResult(node=node, score=score, rationale=rationale))
        return results

    def by_attribute(
        self, key: str, value: object, limit: int = 5
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for node in self._kb.search_by_attribute(key, value)[:limit]:
            results.append(
                RetrievalResult(
                    node=node,
                    score=1.0,
                    rationale=f"matching attribute {key}={value}",
                )
            )
        return results

    def combine(self, *result_sets: Iterable[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """Merge multiple retrieval passes while deduplicating nodes."""

        merged: dict[str, RetrievalResult] = {}
        for result_set in result_sets:
            for result in result_set:
                identifier = result.node.identifier
                existing = merged.get(identifier)
                if not existing or result.score > existing.score:
                    merged[identifier] = result
        ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ordered[:top_k]
