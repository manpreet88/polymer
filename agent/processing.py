"""Utilities for turning raw PSMILES strings into encoder-ready tensors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from Data_Modalities import AdvancedPolymerMultimodalExtractor

from .models import PSMILESDebertaEncoder


@dataclass
class PolymerModalities:
    """Container that stores multimodal representations for a polymer."""

    canonical_psmiles: str
    graph: Optional[Dict] = None
    geometry: Optional[Dict] = None
    fingerprints: Optional[Dict] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> Dict[str, str]:
        features = {}
        if self.graph and "graph_features" in self.graph:
            features.update(self.graph["graph_features"])
        if self.geometry and self.geometry.get("best_conformer"):
            best = self.geometry["best_conformer"]
            features["conformer_energy"] = best.get("energy")
        return {k: v for k, v in features.items() if v is not None}


class PolymerPreprocessor:
    """Thin wrapper around :class:`AdvancedPolymerMultimodalExtractor` for ad-hoc use."""

    def __init__(self) -> None:
        self.extractor = AdvancedPolymerMultimodalExtractor(csv_file="")

    def process(self, psmiles: str, metadata: Optional[Dict[str, str]] = None) -> PolymerModalities:
        canonical = self.extractor.validate_and_standardize_smiles(psmiles)
        if canonical is None:
            raise ValueError("Provided PSMILES string is invalid after sanitization.")
        graph = self.extractor.generate_molecular_graph(canonical)
        geometry = self.extractor.optimize_3d_geometry(canonical)
        fingerprints = self.extractor.calculate_morgan_fingerprints(canonical)
        return PolymerModalities(
            canonical_psmiles=canonical,
            graph=graph or None,
            geometry=geometry or None,
            fingerprints=fingerprints or None,
            metadata=metadata or {},
        )

    def to_encoder_inputs(
        self,
        sample: PolymerModalities,
        ps_encoder: Optional[PSMILESDebertaEncoder] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Convert processed modalities into tensors ready for the encoders."""

        batch: Dict[str, Dict[str, torch.Tensor]] = {}
        if sample.graph:
            node_features: List[Dict] = sample.graph.get("node_features", [])
            z = torch.tensor([f.get("atomic_num", 0) for f in node_features], dtype=torch.long)
            chirality = torch.tensor([float(f.get("chirality", 0.0)) for f in node_features], dtype=torch.float32)
            formal_charge = torch.tensor([float(f.get("formal_charge", 0.0)) for f in node_features], dtype=torch.float32)
            edge_index_pairs = sample.graph.get("edge_indices", [])
            if edge_index_pairs:
                edge_index = torch.tensor(edge_index_pairs, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_features = sample.graph.get("edge_features", [])
            if edge_features:
                edge_attr = torch.tensor(
                    [
                        [
                            float(edge.get("bond_type", 0.0)),
                            float(edge.get("is_aromatic", 0.0)),
                            float(edge.get("is_conjugated", 0.0)),
                        ]
                        for edge in edge_features
                    ],
                    dtype=torch.float32,
                )
            else:
                edge_attr = torch.zeros((0, 3), dtype=torch.float32)
            batch["gine"] = {
                "z": z,
                "chirality": chirality,
                "formal_charge": formal_charge,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "batch": torch.zeros(z.size(0), dtype=torch.long),
            }
        if sample.geometry and sample.geometry.get("best_conformer"):
            conformer = sample.geometry["best_conformer"]
            coords = torch.tensor(conformer.get("coordinates", []), dtype=torch.float32)
            atomic_numbers = torch.tensor(conformer.get("atomic_numbers", []), dtype=torch.long)
            if coords.ndim == 1:
                coords = coords.view(-1, 3)
            batch["schnet"] = {
                "z": atomic_numbers,
                "pos": coords,
                "batch": torch.zeros(atomic_numbers.size(0), dtype=torch.long),
            }
        if sample.fingerprints:
            bits = sample.fingerprints.get("morgan_r3_bits") or sample.fingerprints.get("morgan_r2_bits")
            if bits is not None:
                if isinstance(bits, str):
                    bits_list = [int(ch) for ch in bits]
                else:
                    bits_list = [int(ch) for ch in bits]
                fp_vec = torch.tensor(bits_list, dtype=torch.long)
                batch["fp"] = {
                    "input_ids": fp_vec.unsqueeze(0),
                    "attention_mask": torch.ones_like(fp_vec, dtype=torch.bool).unsqueeze(0),
                }
        if ps_encoder is not None:
            tokenized = ps_encoder.tokenize([sample.canonical_psmiles])
            batch["psmiles"] = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        return batch


__all__ = ["PolymerPreprocessor", "PolymerModalities"]
