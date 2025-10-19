"""Lightweight reimplementation of the multimodal encoders for inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    from torch_geometric.nn import GINEConv
    from torch_geometric.nn.models import SchNet as PyGSchNet
    from torch_geometric.nn import radius_graph
except Exception:  # pragma: no cover - gracefully handle missing torch-geometric
    GINEConv = None
    PyGSchNet = None
    radius_graph = None

try:  # pragma: no cover - optional dependency
    from transformers import DebertaV2Config, DebertaV2ForMaskedLM
except Exception:  # pragma: no cover - fallback when transformers missing
    DebertaV2ForMaskedLM = None
    DebertaV2Config = None

LOGGER = logging.getLogger(__name__)

# Hyperparameters match the training script defaults
MAX_ATOMIC_Z = 85
MASK_ATOM_ID = MAX_ATOMIC_Z + 1
NODE_EMB_DIM = 300
EDGE_EMB_DIM = 300
NUM_GNN_LAYERS = 5
SCHNET_HIDDEN = 600
SCHNET_NUM_INTERACTIONS = 6
SCHNET_NUM_GAUSSIANS = 50
SCHNET_CUTOFF = 10.0
SCHNET_MAX_NEIGHBORS = 64
FP_LENGTH = 2048
FP_HIDDEN = 256
FP_NUM_LAYERS = 4
FP_NUM_HEADS = 8
FP_FF_DIM = 1024
PSMILES_MAX_LEN = 128
DEBERTA_HIDDEN = 600
CONTRASTIVE_EMB_DIM = 600
TEMPERATURE = 0.07


class SimplePSMILESTokenizer:
    """Character-level tokenizer used when no trained tokenizer is available."""

    def __init__(self, max_len: int = PSMILES_MAX_LEN) -> None:
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-=#()[]@+/\\.")
        self.vocab = {ch: idx + 5 for idx, ch in enumerate(alphabet)}
        self.vocab["<pad>"] = 0
        self.vocab["<mask>"] = 1
        self.vocab["<unk>"] = 2
        self.vocab["<cls>"] = 3
        self.vocab["<sep>"] = 4
        self.mask_token = "<mask>"
        self.mask_token_id = self.vocab[self.mask_token]
        self.vocab_size = len(self.vocab)
        self.max_len = max_len

    def __call__(self, text: str, *, truncation: bool = True, padding: str = "max_length", max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        max_len = max_length or self.max_len
        tokens = [self.vocab.get(ch, self.vocab["<unk>"]) for ch in text][:max_len]
        attention = [1] * len(tokens)
        if len(tokens) < max_len:
            pad_length = max_len - len(tokens)
            tokens.extend([self.vocab["<pad>"]] * pad_length)
            attention.extend([0] * pad_length)
        return {"input_ids": tokens, "attention_mask": attention}

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return len(self.vocab)


class GineBlock(nn.Module):
    def __init__(self, node_dim: int) -> None:
        super().__init__()
        if GINEConv is None:
            raise RuntimeError("torch_geometric is required for GINE inference.")
        self.mlp = nn.Sequential(nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim))
        self.conv = GINEConv(self.mlp)
        self.bn = nn.BatchNorm1d(node_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        return self.act(x)


class GineEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.atom_emb = nn.Embedding(num_embeddings=MASK_ATOM_ID + 1, embedding_dim=NODE_EMB_DIM)
        self.node_attr_proj = nn.Sequential(nn.Linear(2, NODE_EMB_DIM), nn.ReLU(), nn.Linear(NODE_EMB_DIM, NODE_EMB_DIM))
        self.edge_encoder = nn.Sequential(nn.Linear(3, EDGE_EMB_DIM), nn.ReLU(), nn.Linear(EDGE_EMB_DIM, EDGE_EMB_DIM))
        self.edge_to_node = nn.Linear(EDGE_EMB_DIM, NODE_EMB_DIM) if EDGE_EMB_DIM != NODE_EMB_DIM else None
        self.layers = nn.ModuleList([GineBlock(NODE_EMB_DIM) for _ in range(NUM_GNN_LAYERS)])
        self.pool_proj = nn.Linear(NODE_EMB_DIM, NODE_EMB_DIM)
        self.node_classifier = nn.Linear(NODE_EMB_DIM, MASK_ATOM_ID + 1)

    def _encode_nodes(self, z: torch.Tensor, chirality: torch.Tensor, charge: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        z = z.to(device)
        chirality = chirality.to(device) if chirality is not None else torch.zeros_like(z, dtype=torch.float, device=device)
        charge = charge.to(device) if charge is not None else torch.zeros_like(z, dtype=torch.float, device=device)
        node_attr = torch.stack([chirality, charge], dim=1).to(device)
        x = self.atom_emb(z) + self.node_attr_proj(node_attr)
        edge_attr = edge_attr.to(device) if edge_attr is not None else torch.zeros((edge_index.size(1), 3), device=device)
        edge_attr = self.edge_encoder(edge_attr)
        if self.edge_to_node is not None:
            edge_attr = self.edge_to_node(edge_attr)
        for layer in self.layers:
            x = layer(x, edge_index.to(device), edge_attr)
        return x

    def forward(self, z: torch.Tensor, chirality: torch.Tensor, charge: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self._encode_nodes(z, chirality, charge, edge_index, edge_attr)
        if batch is None or batch.numel() == 0:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            pooled = torch.zeros((int(batch.max().item()) + 1, h.size(1)), device=h.device)
            for idx in range(pooled.size(0)):
                mask = batch == idx
                if mask.any():
                    pooled[idx] = h[mask].mean(dim=0)
        return self.pool_proj(pooled)

    def node_logits(self, z: torch.Tensor, chirality: torch.Tensor, charge: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self._encode_nodes(z, chirality, charge, edge_index, edge_attr)
        return self.node_classifier(h)


class NodeSchNetWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if PyGSchNet is None:
            raise RuntimeError("torch_geometric is required for SchNet inference.")
        self.schnet = PyGSchNet(
            hidden_channels=SCHNET_HIDDEN,
            num_filters=SCHNET_HIDDEN,
            num_interactions=SCHNET_NUM_INTERACTIONS,
            num_gaussians=SCHNET_NUM_GAUSSIANS,
            cutoff=SCHNET_CUTOFF,
            max_num_neighbors=SCHNET_MAX_NEIGHBORS,
        )
        self.pool_proj = nn.Linear(SCHNET_HIDDEN, SCHNET_HIDDEN)
        self.node_classifier = nn.Linear(SCHNET_HIDDEN, MASK_ATOM_ID + 1)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        z = z.to(device)
        pos = pos.to(device)
        if batch is None or batch.numel() == 0:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=device)
        try:
            edge_index = radius_graph(pos, r=SCHNET_CUTOFF, batch=batch, max_num_neighbors=SCHNET_MAX_NEIGHBORS)
        except Exception:
            edge_index = None
        node_h = self.schnet(z=z, pos=pos, batch=batch)
        if node_h is None:
            raise RuntimeError("SchNet forward returned None.")
        pooled = torch.zeros((int(batch.max().item()) + 1, node_h.size(1)), device=node_h.device)
        for idx in range(pooled.size(0)):
            mask = batch == idx
            if mask.any():
                pooled[idx] = node_h[mask].mean(dim=0)
        return self.pool_proj(pooled)

    def node_logits(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        z = z.to(device)
        pos = pos.to(device)
        if batch is None or batch.numel() == 0:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=device)
        logits = self.schnet(z=z, pos=pos, batch=batch)
        if logits is None:
            raise RuntimeError("SchNet returned None for node logits.")
        return self.node_classifier(logits)


class FingerprintEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(3, FP_HIDDEN)
        self.pos_emb = nn.Embedding(FP_LENGTH, FP_HIDDEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FP_HIDDEN,
            nhead=FP_NUM_HEADS,
            dim_feedforward=FP_FF_DIM,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=FP_NUM_LAYERS)
        self.pool_proj = nn.Linear(FP_HIDDEN, FP_HIDDEN)
        self.token_proj = nn.Linear(FP_HIDDEN, 3)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        pos_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand(input_ids.size(0), -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        key_padding_mask = None if attention_mask is None else ~attention_mask.to(device)
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        if attention_mask is None:
            pooled = out.mean(dim=1)
        else:
            weights = attention_mask.to(device).float().unsqueeze(-1)
            pooled = (out * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.pool_proj(pooled)

    def token_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        pos_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand(input_ids.size(0), -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        key_padding_mask = None if attention_mask is None else ~attention_mask.to(device)
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.token_proj(out)


class PSMILESDebertaEncoder(nn.Module):
    def __init__(self, model_dir: Optional[Path] = None, tokenizer_vocab_size: int = 300) -> None:
        super().__init__()
        if DebertaV2ForMaskedLM is None:
            raise RuntimeError("transformers is required for the PSMILES encoder.")
        config = None
        if DebertaV2Config is not None:
            config = DebertaV2Config(
                vocab_size=tokenizer_vocab_size,
                hidden_size=DEBERTA_HIDDEN,
                num_attention_heads=12,
                num_hidden_layers=12,
                intermediate_size=512,
                pad_token_id=0,
            )
        model_loaded = False
        if model_dir is not None and model_dir.exists():
            try:
                self.model = DebertaV2ForMaskedLM.from_pretrained(str(model_dir))
                model_loaded = True
            except Exception as exc:  # pragma: no cover - missing weights fallback
                LOGGER.warning(
                    "Failed to load pretrained Deberta weights from %s with from_pretrained: %s", model_dir, exc
                )
        if not model_loaded:
            if config is None:
                raise RuntimeError("transformers is installed but DebertaV2Config is unavailable.")
            self.model = DebertaV2ForMaskedLM(config)
        self.pool_proj = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.pool_proj(pooled)

    def token_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        if labels is not None:
            labels = labels.to(device)
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class MultimodalContrastiveModel(nn.Module):
    def __init__(
        self,
        gine: Optional[GineEncoder],
        schnet: Optional[NodeSchNetWrapper],
        fp: Optional[FingerprintEncoder],
        psmiles: Optional[PSMILESDebertaEncoder],
        emb_dim: int = CONTRASTIVE_EMB_DIM,
    ) -> None:
        super().__init__()
        self.gine = gine
        self.schnet = schnet
        self.fp = fp
        self.psmiles = psmiles
        self.temperature = TEMPERATURE
        self.proj_gine = nn.Linear(NODE_EMB_DIM, emb_dim) if gine is not None else None
        self.proj_schnet = nn.Linear(SCHNET_HIDDEN, emb_dim) if schnet is not None else None
        self.proj_fp = nn.Linear(FP_HIDDEN, emb_dim) if fp is not None else None
        self.proj_psmiles = nn.Linear(DEBERTA_HIDDEN, emb_dim) if psmiles is not None else None

    def encode(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        results: Dict[str, torch.Tensor] = {}
        if self.gine is not None and "gine" in batch:
            g = batch["gine"]
            emb = self.gine(g["z"].to(device), g["chirality"].to(device), g["formal_charge"].to(device), g["edge_index"].to(device), g["edge_attr"].to(device), g.get("batch"))
            results["gine"] = F.normalize(self.proj_gine(emb), dim=-1)
        if self.schnet is not None and "schnet" in batch:
            s = batch["schnet"]
            emb = self.schnet(s["z"].to(device), s["pos"].to(device), s.get("batch"))
            results["schnet"] = F.normalize(self.proj_schnet(emb), dim=-1)
        if self.fp is not None and "fp" in batch:
            f = batch["fp"]
            emb = self.fp(f["input_ids"].to(device), f.get("attention_mask"))
            results["fp"] = F.normalize(self.proj_fp(emb), dim=-1)
        if self.psmiles is not None and "psmiles" in batch:
            p = batch["psmiles"]
            emb = self.psmiles(p["input_ids"].to(device), p.get("attention_mask"))
            results["psmiles"] = F.normalize(self.proj_psmiles(emb), dim=-1)
        return results


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


__all__ = [
    "GineEncoder",
    "NodeSchNetWrapper",
    "FingerprintEncoder",
    "PSMILESDebertaEncoder",
    "MultimodalContrastiveModel",
    "SimplePSMILESTokenizer",
    "MAX_ATOMIC_Z",
    "MASK_ATOM_ID",
    "NODE_EMB_DIM",
    "EDGE_EMB_DIM",
    "NUM_GNN_LAYERS",
    "FP_LENGTH",
    "PSMILES_MAX_LEN",
    "CONTRASTIVE_EMB_DIM",
    "count_parameters",
]
