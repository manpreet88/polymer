"""Utilities to run the multimodal encoders on processed polymers."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from .config import CheckpointConfig
from .datamodels import EmbeddingResult, ProcessedPolymer
from .model_loader import load_multimodal_model
from .models import FP_LENGTH

LOGGER = logging.getLogger(__name__)


def _graph_to_tensors(graph: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    if not graph:
        return {
            "z": torch.zeros((0,), dtype=torch.long, device=device),
            "chirality": torch.zeros((0,), dtype=torch.float, device=device),
            "formal_charge": torch.zeros((0,), dtype=torch.float, device=device),
            "edge_index": torch.zeros((2, 0), dtype=torch.long, device=device),
            "edge_attr": torch.zeros((0, 3), dtype=torch.float, device=device),
            "batch": torch.zeros((0,), dtype=torch.long, device=device),
        }
    node_features = graph.get("node_features", [])
    z = torch.tensor([nf.get("atomic_num", 0) for nf in node_features], dtype=torch.long, device=device)
    chirality = torch.tensor([float(nf.get("chirality", 0.0)) for nf in node_features], dtype=torch.float, device=device)
    charges = torch.tensor([float(nf.get("formal_charge", 0.0)) for nf in node_features], dtype=torch.float, device=device)

    edge_indices = graph.get("edge_indices", [])
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_features = graph.get("edge_features", [])
    attrs: List[List[float]] = []
    if edge_index.size(1) > 0:
        for idx in range(edge_index.size(1)):
            feat_idx = min(idx // 2, len(edge_features) - 1) if edge_features else -1
            feat = edge_features[feat_idx] if feat_idx >= 0 else {}
            attrs.append(
                [
                    float(feat.get("bond_type", 0.0)),
                    float(feat.get("is_aromatic", 0.0)),
                    float(feat.get("is_conjugated", 0.0)),
                ]
            )
    edge_attr = torch.tensor(attrs, dtype=torch.float, device=device) if attrs else torch.zeros((edge_index.size(1), 3), dtype=torch.float, device=device)

    batch = torch.zeros((z.size(0),), dtype=torch.long, device=device)
    return {
        "z": z,
        "chirality": chirality,
        "formal_charge": charges,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch,
    }


def _geometry_to_tensors(geometry: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    if not geometry:
        return {
            "z": torch.zeros((0,), dtype=torch.long, device=device),
            "pos": torch.zeros((0, 3), dtype=torch.float, device=device),
            "batch": torch.zeros((0,), dtype=torch.long, device=device),
        }
    best = geometry.get("best_conformer") or {}
    atomic = best.get("atomic_numbers") or geometry.get("atomic_numbers") or []
    coords = best.get("coordinates") or geometry.get("coordinates") or []
    if len(atomic) != len(coords):
        length = min(len(atomic), len(coords))
        atomic = atomic[:length]
        coords = coords[:length]
    z = torch.tensor(atomic, dtype=torch.long, device=device)
    pos = torch.tensor(coords, dtype=torch.float, device=device)
    batch = torch.zeros((z.size(0),), dtype=torch.long, device=device)
    return {"z": z, "pos": pos, "batch": batch}


def _fingerprints_to_tensor(fingerprints: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    if not fingerprints:
        vec = torch.zeros((1, FP_LENGTH), dtype=torch.long, device=device)
    else:
        key = next((k for k in fingerprints if k.endswith("_bits")), None)
        bits = fingerprints.get(key) if key else None
        if bits is None:
            vec = torch.zeros((1, FP_LENGTH), dtype=torch.long, device=device)
        else:
            arr = [1 if b in ("1", 1, True) else 0 for b in bits[:FP_LENGTH]]
            if len(arr) < FP_LENGTH:
                arr.extend([0] * (FP_LENGTH - len(arr)))
            vec = torch.tensor(arr, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(vec, dtype=torch.bool, device=device)
    return {"input_ids": vec, "attention_mask": attn}


def _psmiles_to_tensor(psmiles: str, tokenizer, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(psmiles, truncation=True, padding="max_length", max_length=getattr(tokenizer, "model_max_length", 128))
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class MultimodalEmbeddingService:
    """High-level helper that embeds polymers using the contrastive model."""

    def __init__(
        self,
        checkpoints: Optional[CheckpointConfig] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.checkpoints = (checkpoints or CheckpointConfig()).resolve()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = load_multimodal_model(self.checkpoints, device=self.device)
        self.model.eval()

    def _build_batch(self, polymer: ProcessedPolymer) -> Dict[str, Dict[str, torch.Tensor]]:
        modalities = polymer.modalities
        batch = {}
        batch["gine"] = _graph_to_tensors(modalities.graph, self.device)
        batch["schnet"] = _geometry_to_tensors(modalities.geometry, self.device)
        batch["fp"] = _fingerprints_to_tensor(modalities.fingerprints, self.device)
        batch["psmiles"] = _psmiles_to_tensor(polymer.psmiles, self.tokenizer, self.device)
        return batch

    @torch.inference_mode()
    def embed_polymer(self, polymer: ProcessedPolymer) -> EmbeddingResult:
        """Embed a single processed polymer."""

        batch = self._build_batch(polymer)
        embeddings = self.model.encode(batch)
        fused = np.mean([emb.cpu().numpy() for emb in embeddings.values()], axis=0)
        return EmbeddingResult(vector=fused.squeeze(0), modalities_used=list(embeddings.keys()), reconstruction_losses={})

    @torch.inference_mode()
    def embed_many(self, polymers: Iterable[ProcessedPolymer]) -> List[EmbeddingResult]:
        results: List[EmbeddingResult] = []
        for polymer in polymers:
            try:
                results.append(self.embed_polymer(polymer))
            except Exception as exc:
                LOGGER.warning("Failed to embed %s: %s", polymer.psmiles, exc)
        return results


__all__ = ["MultimodalEmbeddingService"]
