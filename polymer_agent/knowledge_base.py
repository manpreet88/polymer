"""In-memory knowledge graph and persistence helpers for polymer data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import KnowledgeBaseConfig


@dataclass
class KnowledgeNode:
    """Representation of a node in the knowledge graph."""

    identifier: str
    node_type: str
    attributes: Dict[str, object] = field(default_factory=dict)
    neighbors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "identifier": self.identifier,
            "node_type": self.node_type,
            "attributes": self.attributes,
            "neighbors": self.neighbors,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "KnowledgeNode":
        return cls(
            identifier=str(payload["identifier"]),
            node_type=str(payload["node_type"]),
            attributes=dict(payload.get("attributes", {})),
            neighbors=dict(payload.get("neighbors", {})),
        )


class PolymerKnowledgeBase:
    """Simple JSON-backed knowledge graph optimized for RAG workloads."""

    def __init__(self, config: KnowledgeBaseConfig) -> None:
        self._config = config
        self._nodes: Dict[str, KnowledgeNode] = {}
        if config.storage_path and config.storage_path.exists() and config.autoreload:
            self.load(config.storage_path)

    # ------------------------------------------------------------------
    # Node management
    def add_node(self, node: KnowledgeNode) -> None:
        self._nodes[node.identifier] = node

    def get_node(self, identifier: str) -> Optional[KnowledgeNode]:
        return self._nodes.get(identifier)

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        if source not in self._nodes or target not in self._nodes:
            raise KeyError("Both nodes must exist before adding an edge")
        self._nodes[source].neighbors[target] = weight
        self._nodes[target].neighbors[source] = weight

    def iter_nodes(self, node_type: Optional[str] = None) -> Iterable[KnowledgeNode]:
        for node in self._nodes.values():
            if node_type is None or node.node_type == node_type:
                yield node

    # ------------------------------------------------------------------
    # Persistence
    def save(self, path: Optional[Path] = None) -> None:
        path = path or self._config.storage_path
        if not path:
            raise ValueError("No storage path specified for knowledge base persistence")
        payload = {identifier: node.to_dict() for identifier, node in self._nodes.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        self._nodes = {key: KnowledgeNode.from_dict(value) for key, value in payload.items()}

    # ------------------------------------------------------------------
    # Query helpers
    def search_by_attribute(self, key: str, value: object) -> List[KnowledgeNode]:
        return [node for node in self._nodes.values() if node.attributes.get(key) == value]

    def top_neighbors(self, identifier: str, limit: Optional[int] = None) -> List[Tuple[str, float]]:
        node = self._nodes.get(identifier)
        if not node:
            return []
        pairs = sorted(node.neighbors.items(), key=lambda item: item[1], reverse=True)
        if limit is not None:
            pairs = pairs[:limit]
        return pairs

    def add_polymer(self, identifier: str, **attributes: object) -> KnowledgeNode:
        node = KnowledgeNode(identifier=identifier, node_type="polymer", attributes=attributes)
        self.add_node(node)
        return node

    def add_reference(self, identifier: str, **attributes: object) -> KnowledgeNode:
        node = KnowledgeNode(identifier=identifier, node_type="reference", attributes=attributes)
        self.add_node(node)
        return node

    def add_experiment(self, identifier: str, **attributes: object) -> KnowledgeNode:
        node = KnowledgeNode(identifier=identifier, node_type="experiment", attributes=attributes)
        self.add_node(node)
        return node

    def query_similar_embeddings(
        self, embedding: List[float], top_k: int = 5, metric: str = "cosine"
    ) -> List[Tuple[KnowledgeNode, float]]:
        """Return nearest neighbors using cosine or dot-product similarity."""

        if metric not in {"cosine", "dot"}:
            raise ValueError("metric must be 'cosine' or 'dot'")
        results: List[Tuple[KnowledgeNode, float]] = []
        for node in self.iter_nodes("polymer"):
            candidate = node.attributes.get(self._config.embedding_key)
            if not candidate:
                continue
            similarity = _vector_similarity(embedding, candidate, metric)
            results.append((node, similarity))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


def _vector_similarity(a: List[float], b: List[float], metric: str) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match")
    if metric == "dot":
        return sum(x * y for x, y in zip(a, b))
    # cosine similarity
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
