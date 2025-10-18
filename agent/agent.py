"""High level orchestrator that wires retrieval, tools and GPT-4 together."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from openai import OpenAI

from .config import AgentConfig
from .knowledge_base import KnowledgeEntry, PolymerKnowledgeBase
from .models import (
    FingerprintEncoder,
    GineEncoder,
    MultimodalContrastiveModel,
    NodeSchNetWrapper,
    PSMILESDebertaEncoder,
)
from .processing import PolymerModalities, PolymerPreprocessor


LOGGER = logging.getLogger(__name__)


class PolymerRAGAgent:
    """Provides ingestion, retrieval and GPT-4 backed reasoning for polymers."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = PolymerPreprocessor()
        self.knowledge_base = PolymerKnowledgeBase(self.config.retrieval)
        self.client = openai_client or OpenAI()
        self._load_encoders()
        self.config.ensure_cache()

    # ------------------------------------------------------------------
    # Encoder loading utilities
    # ------------------------------------------------------------------
    def _load_encoders(self) -> None:
        checkpoints = self.config.checkpoints
        self.gine: Optional[GineEncoder] = None
        self.schnet: Optional[NodeSchNetWrapper] = None
        self.fp: Optional[FingerprintEncoder] = None
        self.psmiles: Optional[PSMILESDebertaEncoder] = None
        self.multimodal: Optional[MultimodalContrastiveModel] = None

        try:
            self.psmiles = PSMILESDebertaEncoder(
                model_dir_or_name=str(checkpoints.psmlm_dir) if checkpoints.psmlm_dir.exists() else None
            )
            self.psmiles.to(self.device)
            self.psmiles.eval()
        except Exception as exc:  # pragma: no cover - runtime environment check
            LOGGER.warning("PSMILES encoder could not be initialised: %s", exc)
            self.psmiles = None

        if checkpoints.gine_dir.exists():
            try:
                self.gine = GineEncoder().to(self.device)
                state = torch.load(checkpoints.gine_dir / "pytorch_model.bin", map_location=self.device)
                self.gine.load_state_dict(state, strict=False)
                self.gine.eval()
            except Exception as exc:
                LOGGER.warning("Failed loading GINE checkpoint: %s", exc)
                self.gine = None
        else:
            self.gine = None

        if checkpoints.schnet_dir.exists():
            try:
                self.schnet = NodeSchNetWrapper().to(self.device)
                state = torch.load(checkpoints.schnet_dir / "pytorch_model.bin", map_location=self.device)
                self.schnet.load_state_dict(state, strict=False)
                self.schnet.eval()
            except Exception as exc:
                LOGGER.warning("Failed loading SchNet checkpoint: %s", exc)
                self.schnet = None

        if checkpoints.fingerprint_dir.exists():
            try:
                self.fp = FingerprintEncoder().to(self.device)
                state = torch.load(checkpoints.fingerprint_dir / "pytorch_model.bin", map_location=self.device)
                self.fp.load_state_dict(state, strict=False)
                self.fp.eval()
            except Exception as exc:
                LOGGER.warning("Failed loading fingerprint checkpoint: %s", exc)
                self.fp = None

        if any([self.gine, self.schnet, self.fp, self.psmiles]) and checkpoints.multimodal_dir.exists():
            try:
                self.multimodal = MultimodalContrastiveModel(
                    gine_encoder=self.gine,
                    schnet_encoder=self.schnet,
                    fingerprint_encoder=self.fp,
                    psmiles_encoder=self.psmiles,
                ).to(self.device)
                state = torch.load(checkpoints.multimodal_dir / "pytorch_model.bin", map_location=self.device)
                self.multimodal.load_state_dict(state, strict=False)
                self.multimodal.eval()
            except Exception as exc:
                LOGGER.warning("Failed loading multimodal checkpoint: %s", exc)
                self.multimodal = None

    # ------------------------------------------------------------------
    # Ingestion and retrieval
    # ------------------------------------------------------------------
    def ingest(self, identifier: str, psmiles: str, metadata: Optional[Dict[str, str]] = None) -> KnowledgeEntry:
        sample = self.preprocessor.process(psmiles, metadata)
        embeddings = self._encode(sample)
        entry = self.knowledge_base.build_entry(identifier, sample, embeddings)
        self.knowledge_base.add(entry)
        return entry

    def retrieve(self, query_psmiles: str) -> List[Tuple[KnowledgeEntry, float]]:
        sample = self.preprocessor.process(query_psmiles)
        embeddings = self._encode(sample)
        return self.knowledge_base.search(embeddings)

    # ------------------------------------------------------------------
    # GPT-4 orchestration
    # ------------------------------------------------------------------
    def answer(self, question: str, query_psmiles: Optional[str] = None) -> Dict[str, str]:
        context_blocks: List[str] = []
        retrieved: List[Tuple[KnowledgeEntry, float]] = []
        if query_psmiles is not None:
            retrieved = self.retrieve(query_psmiles)
            context_blocks = [entry.as_context_block() + f"\nSimilarity: {score:.3f}" for entry, score in retrieved]
        prompt = self._build_prompt(question, context_blocks)
        completion = self.client.responses.create(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_output_tokens=self.config.llm.max_tokens,
            input=[
                {"role": "system", "content": self.config.llm.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.output_text
        return {
            "answer": content,
            "retrieved": [entry.as_context_block() for entry, _ in retrieved],
        }

    # ------------------------------------------------------------------
    # Embedding utilities
    # ------------------------------------------------------------------
    def _encode(self, sample: PolymerModalities) -> Dict[str, np.ndarray]:
        if not any([self.gine, self.schnet, self.fp, self.psmiles]):
            raise RuntimeError("No encoders are available. Ensure checkpoints are present or transformers installed.")
        batch = self.preprocessor.to_encoder_inputs(sample, ps_encoder=self.psmiles)
        tensors = {}
        for key, value in batch.items():
            tensors[key] = {k: v.to(self.device) for k, v in value.items()}
        embeddings: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            if self.multimodal is not None:
                encoded = self.multimodal.encode(tensors)
                for key, tensor in encoded.items():
                    embeddings[key] = tensor.squeeze(0).detach().cpu().numpy()
            else:
                if self.gine and "gine" in tensors:
                    emb = self.gine(**tensors["gine"])
                    embeddings["gine"] = torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu().numpy()
                if self.schnet and "schnet" in tensors:
                    emb = self.schnet(**tensors["schnet"])
                    embeddings["schnet"] = torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu().numpy()
                if self.fp and "fp" in tensors:
                    emb = self.fp(**tensors["fp"])
                    embeddings["fingerprint"] = torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu().numpy()
                if self.psmiles and "psmiles" in tensors:
                    emb = self.psmiles(**tensors["psmiles"])
                    embeddings["psmiles"] = torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu().numpy()
                if embeddings:
                    stack = np.stack(list(embeddings.values()), axis=0)
                    embeddings["multimodal"] = stack.mean(axis=0)
        return embeddings

    def _build_prompt(self, question: str, context_blocks: Iterable[str]) -> str:
        context = "\n\n".join(context_blocks) if context_blocks else "(no retrieval context)"
        return (
            "Retrieved context:\n"
            f"{context}\n\n"
            f"{self.config.llm.user_prefix}: {question}\n"
            f"{self.config.llm.assistant_prefix}:"
        )


__all__ = ["PolymerRAGAgent"]
