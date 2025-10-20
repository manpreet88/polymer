"""High level helpers that expose polymer workflows as callable services."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

try:
    from .multimodal_components import (
        FingerprintEncoder,
        GineEncoder,
        LoadedMultimodalModel,
        MultimodalContrastiveModel,
        NodeSchNetWrapper,
        PSMILESDebertaEncoder,
        FP_LENGTH,
    )
    _MULTIMODAL_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional heavyweight deps
    FingerprintEncoder = None  # type: ignore
    GineEncoder = None  # type: ignore
    LoadedMultimodalModel = None  # type: ignore
    MultimodalContrastiveModel = None  # type: ignore
    NodeSchNetWrapper = None  # type: ignore
    PSMILESDebertaEncoder = None  # type: ignore
    FP_LENGTH = 2048  # sensible default to keep tokenizer usable
    _MULTIMODAL_IMPORT_ERROR = exc


class SimplePSMILESTokenizer:
    """Character-level tokenizer that mirrors the lightweight helper in ``CL.py``."""

    def __init__(self, max_length: int = 128) -> None:
        self.max_length = max_length
        base_vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-=#()[]@+/\\.")
        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
        self.vocab = self.special_tokens + base_vocab
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.pad_token_id = self.token_to_id["<pad>"]

    @property
    def vocab_size(self) -> int:  # pragma: no cover - trivial property
        return len(self.vocab)

    def _encode_text(self, text: str) -> List[int]:
        tokens = ["<s>"]
        tokens.extend(list(text or ""))
        tokens.append("</s>")
        ids = [self.token_to_id.get(tok, self.token_to_id["<unk>"]) for tok in tokens]
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        else:
            ids.extend([self.pad_token_id] * (self.max_length - len(ids)))
        return ids

    def __call__(self, text: str, *, truncation: bool = True, padding: str = "max_length", max_length: Optional[int] = None) -> Dict[str, List[int]]:
        max_len = max_length or self.max_length
        if max_len != self.max_length:
            # produce dynamic length copy when requested
            original = self.max_length
            self.max_length = max_len
            ids = self._encode_text(text)
            self.max_length = original
        else:
            ids = self._encode_text(text)
        attention = [1 if idx != self.pad_token_id else 0 for idx in ids]
        return {"input_ids": ids, "attention_mask": attention}


@dataclass
class PolymerSample:
    """Container representing one polymer entry with multimodal views."""

    name: str
    psmiles: str
    graph: Dict
    geometry: Dict
    fingerprints: Dict


def _load_json_if_needed(value):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value or {}


def parse_polymer_record(record: Dict) -> PolymerSample:
    """Parse a row coming from the processed CSV into a :class:`PolymerSample`."""

    return PolymerSample(
        name=str(record.get("polymer_id") or record.get("name") or record.get("psmiles") or "polymer"),
        psmiles=str(record.get("psmiles") or record.get("smiles") or ""),
        graph=_load_json_if_needed(record.get("graph")),
        geometry=_load_json_if_needed(record.get("geometry")),
        fingerprints=_load_json_if_needed(record.get("fingerprints")),
    )


def _tensor_from_list(data: Iterable, dtype=torch.float) -> torch.Tensor:
    arr = torch.tensor(list(data), dtype=dtype)
    if arr.dim() == 0:
        arr = arr.unsqueeze(0)
    return arr


def _build_graph_inputs(samples: List[PolymerSample]) -> Optional[Dict[str, torch.Tensor]]:
    node_z: List[torch.Tensor] = []
    node_ch: List[torch.Tensor] = []
    node_fc: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_attr: List[torch.Tensor] = []
    batch = []
    offset = 0
    for batch_idx, sample in enumerate(samples):
        graph = sample.graph or {}
        nodes = graph.get("node_features") or []
        if not nodes:
            continue
        z = torch.tensor([int(node.get("atomic_num", 0)) for node in nodes], dtype=torch.long)
        chirality = torch.tensor([float(node.get("chirality", 0.0)) for node in nodes], dtype=torch.float)
        formal_charge = torch.tensor([float(node.get("formal_charge", 0.0)) for node in nodes], dtype=torch.float)
        node_z.append(z)
        node_ch.append(chirality)
        node_fc.append(formal_charge)
        batch.append(torch.full((z.size(0),), batch_idx, dtype=torch.long))
        edges = graph.get("edge_indices") or []
        features = graph.get("edge_features") or []
        if edges:
            ei = torch.tensor(edges, dtype=torch.long).t().contiguous() + offset
            edge_indices.append(ei)
            feats = []
            for idx, edge in enumerate(edges):
                feat_dict = features[min(idx // 2, len(features) - 1)] if features else {}
                feats.append([
                    float(feat_dict.get("bond_type", 0.0)),
                    float(feat_dict.get("is_aromatic", False)),
                    float(feat_dict.get("is_conjugated", False)),
                ])
            edge_attr.append(torch.tensor(feats, dtype=torch.float))
        offset += z.size(0)
    if not node_z:
        return None
    return {
        "z": torch.cat(node_z),
        "chirality": torch.cat(node_ch),
        "formal_charge": torch.cat(node_fc),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long),
        "edge_attr": torch.cat(edge_attr, dim=0) if edge_attr else torch.empty((0, 3), dtype=torch.float),
        "batch": torch.cat(batch) if batch else torch.tensor([], dtype=torch.long),
    }


def _build_geometry_inputs(samples: List[PolymerSample]) -> Optional[Dict[str, torch.Tensor]]:
    z_all: List[torch.Tensor] = []
    pos_all: List[torch.Tensor] = []
    batch = []
    for idx, sample in enumerate(samples):
        geom = sample.geometry or {}
        best = geom.get("best_conformer") or {}
        coords = best.get("coordinates") or []
        atomic_numbers = best.get("atomic_numbers") or []
        if not coords or not atomic_numbers:
            continue
        pos = torch.tensor(coords, dtype=torch.float)
        z = torch.tensor(atomic_numbers, dtype=torch.long)
        z_all.append(z)
        pos_all.append(pos)
        batch.append(torch.full((z.size(0),), idx, dtype=torch.long))
    if not z_all:
        return None
    return {
        "z": torch.cat(z_all),
        "pos": torch.cat(pos_all),
        "batch": torch.cat(batch) if batch else torch.tensor([], dtype=torch.long),
    }


def _fingerprint_tensor(fingerprint_bits: Iterable[str]) -> torch.Tensor:
    bit_array = np.zeros(FP_LENGTH, dtype=np.int64)
    bits = list(fingerprint_bits)
    for i, bit in enumerate(bits[:FP_LENGTH]):
        bit_array[i] = 1 if str(bit) == "1" else 0
    if len(bits) < FP_LENGTH:
        bit_array[len(bits) : FP_LENGTH] = 0
    return torch.tensor(bit_array, dtype=torch.long)


def _build_fp_inputs(samples: List[PolymerSample]) -> Optional[Dict[str, torch.Tensor]]:
    fps: List[torch.Tensor] = []
    for sample in samples:
        fp_dict = sample.fingerprints or {}
        bits = None
        for key in sorted(fp_dict.keys()):
            if key.endswith("_bits"):
                bits = fp_dict[key]
                break
        if bits is None:
            continue
        fps.append(_fingerprint_tensor(bits))
    if not fps:
        return None
    stacked = torch.stack(fps, dim=0)
    return {"input_ids": stacked, "attention_mask": torch.ones_like(stacked, dtype=torch.bool)}


def _build_psmiles_inputs(samples: List[PolymerSample], tokenizer) -> Dict[str, torch.Tensor]:
    encoded = [tokenizer(sample.psmiles or "", truncation=True, padding="max_length", max_length=128) for sample in samples]
    input_ids = torch.tensor([item["input_ids"] for item in encoded], dtype=torch.long)
    attention = torch.tensor([item["attention_mask"] for item in encoded], dtype=torch.bool)
    return {"input_ids": input_ids, "attention_mask": attention}


def build_multimodal_batch(samples: List[PolymerSample], tokenizer) -> Dict[str, Dict[str, torch.Tensor]]:
    """Convert :class:`PolymerSample` entries into tensors ready for encoding."""

    batch: Dict[str, Dict[str, torch.Tensor]] = {}
    graph_inputs = _build_graph_inputs(samples)
    if graph_inputs:
        batch["gine"] = graph_inputs
    geometry_inputs = _build_geometry_inputs(samples)
    if geometry_inputs:
        batch["schnet"] = geometry_inputs
    fp_inputs = _build_fp_inputs(samples)
    if fp_inputs:
        batch["fp"] = fp_inputs
    batch["psmiles"] = _build_psmiles_inputs(samples, tokenizer)
    return batch


def combine_modal_embeddings(embeddings: Dict[str, torch.Tensor]) -> np.ndarray:
    """Average the available modality embeddings into a single representation."""

    vectors: List[np.ndarray] = []
    for tensor in embeddings.values():
        vec = tensor.detach().cpu().numpy()
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        vectors.append(vec)
    if not vectors:
        raise ValueError("No embeddings available to combine.")
    stacked = np.stack(vectors, axis=0)
    return stacked.mean(axis=0)


DEFAULT_MULTIMODAL_WEIGHTS = Path("multimodal_output/best/pytorch_model.bin")
DEFAULT_GINE_WEIGHTS = Path("gin_output/best/pytorch_model.bin")
DEFAULT_SCHNET_WEIGHTS = Path("schnet_output/best/pytorch_model.bin")
DEFAULT_FP_WEIGHTS = Path("fingerprint_mlm_output/best/pytorch_model.bin")
DEFAULT_PSMILES_DIR = Path("polybert_output/best")


def load_contrastive_model(
    *,
    multimodal_weights: Path = DEFAULT_MULTIMODAL_WEIGHTS,
    gine_weights: Path = DEFAULT_GINE_WEIGHTS,
    schnet_weights: Path = DEFAULT_SCHNET_WEIGHTS,
    fp_weights: Path = DEFAULT_FP_WEIGHTS,
    psmiles_dir: Path = DEFAULT_PSMILES_DIR,
    device: Optional[str] = None,
) -> LoadedMultimodalModel:
    """Instantiate :class:`MultimodalContrastiveModel` and load available weights."""

    if _MULTIMODAL_IMPORT_ERROR is not None or MultimodalContrastiveModel is None:
        raise RuntimeError(
            "Multimodal encoder components are unavailable. Install transformers/torch geometric dependencies."
        ) from _MULTIMODAL_IMPORT_ERROR

    device_t = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gine = GineEncoder()
    if gine_weights.exists():
        gine.load_state_dict(torch.load(gine_weights, map_location="cpu"), strict=False)

    schnet = None
    try:
        schnet = NodeSchNetWrapper()
        if schnet_weights.exists():
            schnet.load_state_dict(torch.load(schnet_weights, map_location="cpu"), strict=False)
    except RuntimeError:
        schnet = None

    fp = FingerprintEncoder()
    if fp_weights.exists():
        fp.load_state_dict(torch.load(fp_weights, map_location="cpu"), strict=False)

    psmiles = PSMILESDebertaEncoder(model_dir_or_name=str(psmiles_dir) if psmiles_dir.exists() else None)

    model = MultimodalContrastiveModel(gine, schnet, fp, psmiles)
    if multimodal_weights.exists():
        state = torch.load(multimodal_weights, map_location="cpu")
        model.load_state_dict(state, strict=False)

    return LoadedMultimodalModel(model=model, device=device_t)


def encode_polymers(samples: List[PolymerSample], tokenizer, loader: LoadedMultimodalModel) -> Dict[str, torch.Tensor]:
    """Encode polymers into the shared CL embedding space."""

    batch = build_multimodal_batch(samples, tokenizer)
    return loader.encode(batch)
