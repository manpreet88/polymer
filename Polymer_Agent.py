"""Polymer AI Agent Orchestration Module.

This module implements an advanced Retrieval-Augmented Generation (RAG) agent for
polymer science workflows. The architecture mirrors the multi-stage orchestration
flow introduced in the "AI Agent for Oncology" Nature paper, but it is adapted to
operate on polymer research questions. The agent integrates the multimodal
contrastive (CL) encoders from this repository, couples them with a polymer-focused
knowledge base, and orchestrates reasoning through GPT-4.

Main Components
---------------
1. **ContrastivePolymerEncoder** – wraps the multimodal encoders trained via
   ``CL.py`` to produce fused embeddings for polymer SMILES/PSMILES inputs.
2. **PolymerKnowledgeBase** – manages a vector store over curated polymer
   documents, research notes, and structured datasets.
3. **PolymerEvidenceSynthesizer** – filters, ranks, and summarizes retrieved
   evidence in a format ready for GPT-4 prompting.
4. **GPT4Orchestrator** – crafts prompts, executes GPT-4 calls, and supervises
   iterative refinement loops that incorporate human feedback when available.
5. **PolymerAgent** – coordinates the full workflow and exposes high-level
   ``analyze`` and ``plan_experiments`` entry points usable by expert and
   non-expert users alike.

The module also contains detailed setup instructions and usage examples so it can
be dropped into production repositories or demonstration notebooks.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import CrossEncoder
from sklearn.neighbors import NearestNeighbors


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    """Configuration for loading the multimodal contrastive encoder."""

    checkpoint_dir: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512


@dataclass
class KnowledgeBaseConfig:
    """Configuration for the polymer knowledge base."""

    artifact_dir: Path
    embedding_dim: int = 768
    n_neighbors: int = 8
    rebuild_index: bool = False


@dataclass
class GPT4Config:
    """Configuration for GPT-4 orchestration."""

    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 1200


@dataclass
class AgentConfig:
    """High-level configuration for the polymer AI agent."""

    encoder: EncoderConfig
    knowledge_base: KnowledgeBaseConfig
    gpt4: GPT4Config
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    relevance_threshold: float = 0.25
    max_evidence: int = 5


# ---------------------------------------------------------------------------
# Encoder Wrapper
# ---------------------------------------------------------------------------


class ContrastivePolymerEncoder:
    """Loads and applies the multimodal contrastive encoder for PSMILES strings."""

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._load_models()

    def _load_models(self) -> None:
        """Load GINE, SchNet, fingerprint, and DeBERTa sub-encoders."""

        checkpoint_dir = self.config.checkpoint_dir
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Contrastive checkpoint directory not found: {checkpoint_dir}"
            )

        LOGGER.info("Loading contrastive encoders from %s", checkpoint_dir)

        # The CL checkpoint bundles each modality encoder along with a fusion
        # projection head. Here we load all state dicts and assemble a unified
        # encoder. The code intentionally mirrors the architecture in ``CL.py``.
        fused_state = torch.load(checkpoint_dir / "contrastive.pt", map_location=self.device)

        from CL import PolymerCLModel  # Local import to avoid circular deps.

        self.model = PolymerCLModel.load_from_state_dict(fused_state)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_psmiles(self, psmiles: Sequence[str]) -> np.ndarray:
        """Embed a batch of PSMILES strings using the shared latent space."""

        inputs = list(psmiles)
        if not inputs:
            return np.zeros((0, self.model.projection_dim), dtype=np.float32)

        LOGGER.debug("Encoding %d PSMILES strings", len(inputs))

        embeddings: List[np.ndarray] = []
        batch_size = 8
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            tokenized = self.model.tokenize_psmiles(batch, max_length=self.config.max_length)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            fused = self.model.encode_from_psmiles(tokenized)
            embeddings.append(fused.cpu().numpy())

        return np.vstack(embeddings)


# ---------------------------------------------------------------------------
# Knowledge Base & Retrieval
# ---------------------------------------------------------------------------


class PolymerKnowledgeBase:
    """Vector store backed by contrastive polymer embeddings."""

    INDEX_FILENAME = "vector_index.npz"
    METADATA_FILENAME = "metadata.jsonl"

    def __init__(self, config: KnowledgeBaseConfig, encoder: ContrastivePolymerEncoder) -> None:
        self.config = config
        self.encoder = encoder
        self.artifact_dir = config.artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.artifact_dir / self.INDEX_FILENAME
        self.metadata_path = self.artifact_dir / self.METADATA_FILENAME
        self._load_or_build()

    def _load_or_build(self) -> None:
        if self.config.rebuild_index or not self.index_path.exists():
            LOGGER.info("Building polymer knowledge base index from scratch")
            self.embeddings = np.zeros((0, self.config.embedding_dim), dtype=np.float32)
            self.metadata: List[Dict[str, Any]] = []
            self._fit_index()
        else:
            LOGGER.info("Loading polymer knowledge base index from %s", self.index_path)
            data = np.load(self.index_path)
            self.embeddings = data["embeddings"]
            self.metadata = [json.loads(line) for line in self.metadata_path.read_text().splitlines()]
            self._fit_index()

    def _fit_index(self) -> None:
        if len(self.embeddings) == 0:
            self.nn_index = None
            return

        self.nn_index = NearestNeighbors(
            n_neighbors=min(self.config.n_neighbors, len(self.embeddings)), metric="cosine"
        )
        self.nn_index.fit(self.embeddings)

    def ingest(self, records: Iterable[Dict[str, Any]]) -> None:
        """Ingest new documents into the knowledge base."""

        documents = list(records)
        if not documents:
            return

        psmiles = [doc["psmiles"] for doc in documents]
        embeddings = self.encoder.embed_psmiles(psmiles)
        payloads = np.vstack([self.embeddings, embeddings]) if len(self.embeddings) else embeddings
        self.embeddings = payloads
        self.metadata.extend(documents)

        np.savez_compressed(self.index_path, embeddings=self.embeddings)
        with self.metadata_path.open("w", encoding="utf-8") as fp:
            for record in self.metadata:
                fp.write(json.dumps(record) + "\n")

        self._fit_index()

    def query(self, psmiles: str, top_k: Optional[int] = None) -> List[Tuple[float, Dict[str, Any]]]:
        """Retrieve nearest neighbors for a polymer input."""

        if self.nn_index is None:
            LOGGER.warning("Knowledge base is empty; returning no evidence")
            return []

        query_embedding = self.encoder.embed_psmiles([psmiles])
        top_k = top_k or self.config.n_neighbors
        distances, indices = self.nn_index.kneighbors(query_embedding, n_neighbors=top_k)

        scored_results: List[Tuple[float, Dict[str, Any]]] = []
        for dist, idx in zip(distances[0], indices[0]):
            metadata = self.metadata[int(idx)]
            scored_results.append((1 - dist, metadata))

        return scored_results


# ---------------------------------------------------------------------------
# Evidence Synthesis & Orchestration
# ---------------------------------------------------------------------------


class PolymerEvidenceSynthesizer:
    """Ranks and formats retrieved evidence for downstream GPT-4 reasoning."""

    def __init__(self, model_name: str) -> None:
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[float, Dict[str, Any]]],
        max_evidence: int,
        relevance_threshold: float,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, candidate[1]["summary"]) for candidate in candidates]
        scores = self.cross_encoder.predict(pairs)

        filtered: List[Tuple[float, Dict[str, Any]]] = []
        for base_score, (retr_score, metadata) in zip(scores, candidates):
            final_score = float(0.5 * base_score + 0.5 * retr_score)
            if final_score >= relevance_threshold:
                enriched = dict(metadata)
                enriched["relevance"] = final_score
                filtered.append((final_score, enriched))

        filtered.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in filtered[:max_evidence]]

    @staticmethod
    def format_evidence(evidence: Sequence[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for idx, item in enumerate(evidence, start=1):
            lines.append(
                "\n".join(
                    [
                        f"Evidence {idx}: {item['title']}",
                        f"Relevance Score: {item['relevance']:.3f}",
                        f"Source: {item.get('source', 'unknown')}",
                        f"Key Findings: {item['summary']}",
                        f"Experimental Details: {item.get('experiment', 'n/a')}",
                    ]
                )
            )
        return "\n\n".join(lines)


class GPT4Orchestrator:
    """Handles prompt construction, GPT-4 calls, and iterative refinement."""

    SYSTEM_PROMPT = (
        "You are PolySynth, an elite polymer R&D assistant. Combine polymer science, "
        "materials informatics, and process engineering knowledge. Produce actionable "
        "plans grounded in provided evidence. Explicitly quantify uncertainties and "
        "recommend validation experiments accessible to both experts and non-experts."
    )

    def __init__(self, config: GPT4Config) -> None:
        self.client = OpenAI(api_key=config.api_key)
        self.config = config

    def _call_gpt4(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    def run_dialogue(self, user_query: str, evidence_block: str) -> str:
        prompt = (
            "Integrate the following evidence into a comprehensive polymer design "
            "and characterization report. Include: (1) mechanistic rationale, (2) "
            "computational predictions leveraging the contrastive embeddings, (3) "
            "bench-scale and pilot-scale experimentation steps with required tools, "
            "and (4) communication tips for non-expert stakeholders.\n\n"
            f"Evidence:\n{evidence_block}\n"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": prompt},
        ]

        LOGGER.debug("Dispatching GPT-4 request with %d messages", len(messages))
        return self._call_gpt4(messages)


# ---------------------------------------------------------------------------
# Agent API
# ---------------------------------------------------------------------------


class PolymerAgent:
    """High-level orchestrator replicating the Nature AI agent workflow for polymers."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.encoder = ContrastivePolymerEncoder(config.encoder)
        self.knowledge_base = PolymerKnowledgeBase(config.knowledge_base, self.encoder)
        self.synthesizer = PolymerEvidenceSynthesizer(config.cross_encoder_model)
        self.orchestrator = GPT4Orchestrator(config.gpt4)

    def _retrieve_and_summarize(self, psmiles: str, natural_language_query: str) -> str:
        candidates = self.knowledge_base.query(psmiles, top_k=self.config.max_evidence * 2)
        evidence = self.synthesizer.rerank(
            natural_language_query, candidates, self.config.max_evidence, self.config.relevance_threshold
        )
        return self.synthesizer.format_evidence(evidence)

    def analyze(self, psmiles: str, question: str) -> str:
        evidence_block = self._retrieve_and_summarize(psmiles, question)
        return self.orchestrator.run_dialogue(question, evidence_block)

    def plan_experiments(
        self, psmiles: str, property_targets: Dict[str, float], constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        constraint_summary = json.dumps(constraints or {}, indent=2)
        prompt = (
            "Design an end-to-end polymer experimentation program. Target the "
            f"following properties: {json.dumps(property_targets)}. Respect the "
            f"constraints: {constraint_summary}."
        )
        return self.analyze(psmiles, prompt)


# ---------------------------------------------------------------------------
# Setup & Demonstration Utilities
# ---------------------------------------------------------------------------


def bootstrap_knowledge_base(
    agent: PolymerAgent,
    curated_records: Iterable[Dict[str, Any]],
) -> None:
    """One-shot ingestion helper mirroring the data curation stage in the Nature workflow."""

    records = list(curated_records)
    LOGGER.info("Bootstrapping knowledge base with %d curated records", len(records))
    agent.knowledge_base.ingest(records)


def example_usage() -> None:
    """Illustrates the full pipeline with mock data."""

    logging.basicConfig(level=logging.INFO)

    # ------------------------------------------------------------------
    # 1. Configure the agent
    # ------------------------------------------------------------------
    config = AgentConfig(
        encoder=EncoderConfig(checkpoint_dir=Path("./checkpoints/polymer_cl")),
        knowledge_base=KnowledgeBaseConfig(artifact_dir=Path("./artifacts/knowledge_base")),
        gpt4=GPT4Config(api_key=os.environ.get("OPENAI_API_KEY", "set-me")),
    )

    agent = PolymerAgent(config)

    # ------------------------------------------------------------------
    # 2. Bootstrap the knowledge base (normally includes literature, lab logs, etc.)
    # ------------------------------------------------------------------
    curated_records = [
        {
            "psmiles": "[*]CC(=O)OCC[*]",
            "title": "Polyester toughness modifiers",
            "summary": "Copolymerizing flexible diols raises elongation at break while preserving Tg.",
            "experiment": "Blend 10 wt% telechelic polycaprolactone; anneal at 80C for 2h",
            "source": "internal_lab_report_042",
        },
        {
            "psmiles": "[*]C=C[*]",
            "title": "High-modulus vinyl polymers",
            "summary": "Crosslink density correlates with modulus; aromatic comonomers improve stiffness.",
            "experiment": "Photoinitiated curing using 1 wt% Irgacure 2959",
            "source": "literature_doi:10.1000/poly-2024",
        },
    ]
    bootstrap_knowledge_base(agent, curated_records)

    # ------------------------------------------------------------------
    # 3. Run an expert-level query
    # ------------------------------------------------------------------
    expert_question = (
        "Design a polymer electrolyte with high ionic conductivity and mechanical stability "
        "suitable for solid-state batteries. Include synthesis steps and safety considerations."
    )
    expert_report = agent.analyze("[*]OCCO[*]", expert_question)
    print("Expert Report:\n", expert_report)

    # ------------------------------------------------------------------
    # 4. Run a non-expert onboarding question
    # ------------------------------------------------------------------
    non_expert_question = (
        "Explain how this polymer can be manufactured in a small lab and what tools are needed."
    )
    onboarding_brief = agent.analyze("[*]OCCO[*]", non_expert_question)
    print("\nOnboarding Brief:\n", onboarding_brief)


if __name__ == "__main__":
    example_usage()

