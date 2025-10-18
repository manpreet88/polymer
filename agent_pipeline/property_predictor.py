"""Downstream property prediction tools built on top of the multimodal embeddings."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import CheckpointConfig
from .datamodels import EmbeddingResult, ProcessedPolymer, PropertyPrediction
from .embedding_service import MultimodalEmbeddingService

LOGGER = logging.getLogger(__name__)


@dataclass
class PropertyHeadSpec:
    name: str
    unit: Optional[str] = None
    description: Optional[str] = None


class PropertyHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PropertyPredictorEnsemble:
    """Loads property-specific ensembles trained on top of the contrastive embeddings."""

    def __init__(
        self,
        checkpoints: Optional[CheckpointConfig] = None,
        *,
        embedding_service: Optional[MultimodalEmbeddingService] = None,
        device: Optional[torch.device] = None,
        embedding_dim: int = 600,
    ) -> None:
        self.checkpoints = (checkpoints or CheckpointConfig()).resolve()
        self.embedding_service = embedding_service or MultimodalEmbeddingService(self.checkpoints)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.heads: Dict[str, List[PropertyHead]] = {}
        self.specs: Dict[str, PropertyHeadSpec] = {}
        self._load_heads()

    def _load_heads(self) -> None:
        root = self.checkpoints.property_heads_dir
        if not root.exists():
            LOGGER.info("Property head directory %s does not exist; using empty ensemble.", root)
            return
        for prop_dir in root.iterdir():
            if not prop_dir.is_dir():
                continue
            name = prop_dir.name
            metadata_path = prop_dir / "metadata.json"
            unit = None
            description = None
            if metadata_path.exists():
                try:
                    meta = json.loads(metadata_path.read_text())
                    unit = meta.get("unit")
                    description = meta.get("description")
                except Exception as exc:
                    LOGGER.warning("Failed to parse metadata for %s: %s", name, exc)
            ensemble: List[PropertyHead] = []
            for weight_file in sorted(prop_dir.glob("*.pt")):
                try:
                    state = torch.load(weight_file, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state_dict = state["state_dict"]
                    else:
                        state_dict = state
                    head = PropertyHead(self.embedding_dim)
                    head.load_state_dict(state_dict, strict=False)
                    head.to(self.device)
                    head.eval()
                    ensemble.append(head)
                    LOGGER.info("Loaded property head %s from %s", name, weight_file)
                except Exception as exc:
                    LOGGER.warning("Could not load head for %s from %s: %s", name, weight_file, exc)
            if ensemble:
                self.heads[name] = ensemble
                self.specs[name] = PropertyHeadSpec(name=name, unit=unit, description=description)

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------
    def available_properties(self) -> List[str]:
        return sorted(self.heads.keys())

    @torch.inference_mode()
    def predict_from_embedding(self, embedding: EmbeddingResult) -> List[PropertyPrediction]:
        if not self.heads:
            return []
        vector = torch.tensor(embedding.vector, dtype=torch.float32, device=self.device)
        preds: List[PropertyPrediction] = []
        for name, ensemble in self.heads.items():
            outputs = []
            for head in ensemble:
                value = head(vector.unsqueeze(0)).item()
                outputs.append(value)
            mean = float(np.mean(outputs))
            std = float(np.std(outputs))
            spec = self.specs.get(name, PropertyHeadSpec(name))
            preds.append(
                PropertyPrediction(
                    name=name,
                    mean=mean,
                    std=std,
                    unit=spec.unit,
                    raw_outputs={f"model_{idx}": val for idx, val in enumerate(outputs)},
                )
            )
        return preds

    @torch.inference_mode()
    def predict_polymer(self, polymer: ProcessedPolymer) -> List[PropertyPrediction]:
        embedding = self.embedding_service.embed_polymer(polymer)
        return self.predict_from_embedding(embedding)

    @torch.inference_mode()
    def predict_many(self, polymers: Iterable[ProcessedPolymer]) -> Dict[str, List[PropertyPrediction]]:
        results: Dict[str, List[PropertyPrediction]] = {}
        for polymer in polymers:
            try:
                results[polymer.psmiles] = self.predict_polymer(polymer)
            except Exception as exc:
                LOGGER.warning("Failed to predict properties for %s: %s", polymer.psmiles, exc)
        return results


__all__ = ["PropertyPredictorEnsemble", "PropertyHeadSpec"]
