"""Core orchestration logic for the autonomous polymer agent."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .config import AgentConfig
from .data import MultimodalDataBundle, MultimodalDataExtractor
from .knowledge_base import PolymerKnowledgeBase
from .retrieval import HybridRetriever, RetrievalResult
from .visualization import format_table, render_metadata, render_retrieval_results, TableColumn


class AgentMode(str, Enum):
    """Interaction styles for different user archetypes."""

    NON_EXPERT = "non_expert"
    EXPERT = "expert"


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class AgentState:
    """Mutable state tracked across interactions."""

    history: List[ConversationTurn] = field(default_factory=list)
    retrieved_context: List[RetrievalResult] = field(default_factory=list)

    def add_turn(self, role: str, content: str, max_history: int) -> None:
        self.history.append(ConversationTurn(role=role, content=content))
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]


class PolymerAgentOrchestrator:
    """High-level task planner coordinating tools, retrieval, and UI."""

    def __init__(
        self,
        config: AgentConfig,
        knowledge_base: PolymerKnowledgeBase,
        retriever: HybridRetriever,
        extractor: Optional[MultimodalDataExtractor] = None,
    ) -> None:
        self._config = config
        self._kb = knowledge_base
        self._retriever = retriever
        self._extractor = extractor
        self._state = AgentState()

    # ------------------------------------------------------------------
    # Public API
    def handle_query(self, query: str, mode: AgentMode = AgentMode.NON_EXPERT) -> str:
        """Process a free-form question and return an answer template."""

        self._state.add_turn("user", query, self._config.max_history)
        context = self._retrieve_context(query)
        response = self._draft_response(query, context, mode)
        self._state.add_turn("assistant", response, self._config.max_history)
        return response

    def evaluate_candidate(self, bundle: MultimodalDataBundle, mode: AgentMode) -> str:
        """Produce an evaluation summary for a candidate polymer."""

        metadata = render_metadata(bundle.summary())
        context = self._state.retrieved_context
        retrieved = render_retrieval_results(context)
        table = format_table(
            [
                TableColumn("Channel", ["Graph", "Geometry", "Fingerprint"]),
                TableColumn(
                    "Available",
                    [
                        bool(bundle.graph),
                        bool(bundle.geometry),
                        bool(bundle.fingerprint),
                    ],
                ),
            ]
        )
        persona_hint = (
            "Focus on plain-language explanations and highlight next steps."
            if mode == AgentMode.NON_EXPERT
            else "Include technical diagnostics, uncertainties, and retraining hooks."
        )
        return (
            f"Evaluation Persona Hint: {persona_hint}\n"
            f"Multimodal Coverage\n{table}\n\n"
            f"Metadata\n{metadata}\n\n"
            f"Retrieved Context\n{retrieved}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    def _retrieve_context(self, query: str) -> List[RetrievalResult]:
        """Retrieve supporting knowledge base entries for the query."""

        results: List[RetrievalResult] = []
        if self._state.retrieved_context:
            history_context = self._state.retrieved_context
            results.extend(history_context)
        attribute_hits = self._retriever.by_attribute("tag", query.lower())
        embedding_hits: List[RetrievalResult] = []
        if self._extractor is not None and "C" in query:
            try:
                bundle = self._extractor.from_psmiles(query)
            except Exception:
                bundle = None
            if bundle and bundle.metadata.get("embedding"):
                embedding_hits = self._retriever.by_embedding(bundle.metadata["embedding"])
        combined = self._retriever.combine(results, attribute_hits, embedding_hits)
        self._state.retrieved_context = combined
        return combined

    def _draft_response(
        self, query: str, context: List[RetrievalResult], mode: AgentMode
    ) -> str:
        """Synthesize a textual answer template."""

        preamble = self._build_preamble(mode)
        bullets = [
            f"User query: {query}",
            "Contextual evidence:",
            render_retrieval_results(context),
            "Recommended next actions differ for non-experts vs experts.",
        ]
        return "\n".join(itertools.chain([preamble, ""], bullets))

    def _build_preamble(self, mode: AgentMode) -> str:
        if mode == AgentMode.NON_EXPERT:
            return (
                "I will translate polymer design insights into approachable language, "
                "with visuals and experimental suggestions tailored to newcomers."
            )
        return (
            "I will provide a technical roadmap covering encoder reuse, fine-tuning, "
            "and validation analytics for expert users."
        )
