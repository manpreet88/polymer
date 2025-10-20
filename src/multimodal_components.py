"""Lightweight re-implementation of the multimodal encoders used in CL.py.

The goal of this module is to provide the minimal network components needed for
inference so that we can safely import them without triggering the heavy
training pipelines contained in the original scripts.  The definitions mirror
the corresponding classes in ``CL.py`` but drop training-specific utilities.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Torch Geometric is optional during unit tests
    from torch_geometric.nn import GINEConv
    from torch_geometric.nn.models import SchNet as PyGSchNet
    from torch_geometric.nn import radius_graph
except Exception:  # pragma: no cover - handled gracefully at runtime
    GINEConv = None
    PyGSchNet = None
    radius_graph = None

from transformers import DebertaV2ForMaskedLM, DebertaV2Config

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
VOCAB_SIZE_FP = 3
DEBERTA_HIDDEN = 600
TEMPERATURE = 0.07


def _ensure_available(module, name: str) -> None:
    if module is None:
        raise RuntimeError(
            f"Required dependency '{name}' is not available. Install torch-geometric"
            " with the appropriate extras before using the contrastive tools."
        )


class GineBlock(nn.Module):
    """Single GINE layer with residual/normalisation just like in CL.py."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        _ensure_available(GINEConv, "torch_geometric")
        nn_lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv = GINEConv(nn_lin)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(x + F.relu(h))
        return h


class GineEncoder(nn.Module):
    """Graph encoder matching the architecture used during pre-training."""

    def __init__(
        self,
        node_emb_dim: int = NODE_EMB_DIM,
        edge_emb_dim: int = EDGE_EMB_DIM,
        num_layers: int = NUM_GNN_LAYERS,
    ) -> None:
        super().__init__()
        _ensure_available(GINEConv, "torch_geometric")
        self.atom_emb = nn.Embedding(MASK_ATOM_ID + 1, node_emb_dim)
        self.node_attr_proj = nn.Sequential(
            nn.Linear(2, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, node_emb_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim),
        )
        self.edge_to_node = (
            nn.Linear(edge_emb_dim, node_emb_dim)
            if edge_emb_dim != node_emb_dim
            else nn.Identity()
        )
        self.layers = nn.ModuleList([GineBlock(node_emb_dim) for _ in range(num_layers)])
        self.pool_proj = nn.Linear(node_emb_dim, node_emb_dim)

    def _compute_node_reps(
        self,
        z: torch.Tensor,
        chirality: torch.Tensor,
        formal_charge: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        atom_emb = self.atom_emb(z.to(device))
        if chirality is None or formal_charge is None:
            node_attr = torch.zeros((z.size(0), 2), device=device)
        else:
            node_attr = torch.stack([chirality, formal_charge], dim=1).to(device)
        node_attr_emb = self.node_attr_proj(node_attr)
        x = atom_emb + node_attr_emb
        if edge_attr is None or edge_attr.numel() == 0:
            edge_emb = torch.zeros((0, EDGE_EMB_DIM), dtype=torch.float, device=device)
        else:
            edge_emb = self.edge_encoder(edge_attr.to(device))
        if not isinstance(self.edge_to_node, nn.Identity) and edge_emb.numel() > 0:
            edge_emb = self.edge_to_node(edge_emb)
        for layer in self.layers:
            x = layer(x, edge_index.to(device), edge_emb)
        return x

    def forward(
        self,
        z: torch.Tensor,
        chirality: torch.Tensor,
        formal_charge: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self._compute_node_reps(z, chirality, formal_charge, edge_index, edge_attr)
        if batch is None or batch.numel() == 0:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            pooled = torch.zeros((int(batch.max().item()) + 1, h.size(1)), device=h.device)
            for idx in range(pooled.size(0)):
                mask = batch == idx
                if mask.any():
                    pooled[idx] = h[mask].mean(dim=0)
        return self.pool_proj(pooled)


class NodeSchNetWrapper(nn.Module):
    """Wrapper around torch-geometric's SchNet encoder."""

    def __init__(
        self,
        hidden_channels: int = SCHNET_HIDDEN,
        num_interactions: int = SCHNET_NUM_INTERACTIONS,
        num_gaussians: int = SCHNET_NUM_GAUSSIANS,
        cutoff: float = SCHNET_CUTOFF,
        max_num_neighbors: int = SCHNET_MAX_NEIGHBORS,
    ) -> None:
        super().__init__()
        _ensure_available(PyGSchNet, "torch_geometric")
        self.schnet = PyGSchNet(
            hidden_channels=hidden_channels,
            num_filters=hidden_channels,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
        )
        self.pool_proj = nn.Linear(hidden_channels, hidden_channels)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        z = z.to(device)
        pos = pos.to(device)
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=device)
        else:
            batch = batch.to(device)
        try:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        except Exception:
            edge_index = None
        if edge_index is not None and edge_index.numel() > 0:
            out = self.schnet(z=z, pos=pos, edge_index=edge_index, batch=batch)
        else:
            out = self.schnet(z=z, pos=pos, batch=batch)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        if batch.numel() == 0:
            pooled = out.mean(dim=0, keepdim=True)
        else:
            pooled = torch.zeros((int(batch.max().item()) + 1, out.size(-1)), device=out.device)
            for idx in range(pooled.size(0)):
                mask = batch == idx
                if mask.any():
                    pooled[idx] = out[mask].mean(dim=0)
        return self.pool_proj(pooled)


class FingerprintEncoder(nn.Module):
    """Small Transformer encoder over binary fingerprints."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE_FP,
        hidden_dim: int = 256,
        seq_len: int = FP_LENGTH,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(device)
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        pooled = out.mean(dim=1)
        return self.pool_proj(pooled)


class PSMILESDebertaEncoder(nn.Module):
    """Wrapper around DeBERTa used for polymer sequence encoding."""

    def __init__(self, model_dir_or_name: Optional[str] = None) -> None:
        super().__init__()
        try:
            if model_dir_or_name and os.path.isdir(model_dir_or_name):
                self.model = DebertaV2ForMaskedLM.from_pretrained(model_dir_or_name)
            else:
                self.model = DebertaV2ForMaskedLM.from_pretrained("microsoft/deberta-v2-xlarge")
        except Exception:
            config = DebertaV2Config(
                vocab_size=30522,
                hidden_size=DEBERTA_HIDDEN,
                num_attention_heads=12,
                num_hidden_layers=12,
                intermediate_size=4 * DEBERTA_HIDDEN,
            )
            self.model = DebertaV2ForMaskedLM(config)
        self.pool_proj = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).to(hidden.device).float()
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.pool_proj(pooled)


class MultimodalContrastiveModel(nn.Module):
    """Contrastive backbone that exposes an ``encode`` helper used by the tools."""

    def __init__(
        self,
        gine_encoder: Optional[GineEncoder],
        schnet_encoder: Optional[NodeSchNetWrapper],
        fp_encoder: Optional[FingerprintEncoder],
        psmiles_encoder: Optional[PSMILESDebertaEncoder],
        emb_dim: int = 600,
    ) -> None:
        super().__init__()
        self.gine = gine_encoder
        self.schnet = schnet_encoder
        self.fp = fp_encoder
        self.psmiles = psmiles_encoder
        self.proj_gine = (
            nn.Linear(gine_encoder.pool_proj.out_features, emb_dim) if gine_encoder else None
        )
        self.proj_schnet = (
            nn.Linear(schnet_encoder.pool_proj.out_features, emb_dim) if schnet_encoder else None
        )
        self.proj_fp = (
            nn.Linear(fp_encoder.pool_proj.out_features, emb_dim) if fp_encoder else None
        )
        self.proj_psmiles = (
            nn.Linear(psmiles_encoder.pool_proj.out_features, emb_dim) if psmiles_encoder else None
        )
        self.temperature = TEMPERATURE

    def encode(self, batch_mods: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self.gine is not None and "gine" in batch_mods:
            emb = self.gine(**batch_mods["gine"])
            out["gine"] = F.normalize(self.proj_gine(emb), dim=-1)
        if self.schnet is not None and "schnet" in batch_mods:
            emb = self.schnet(**batch_mods["schnet"])
            out["schnet"] = F.normalize(self.proj_schnet(emb), dim=-1)
        if self.fp is not None and "fp" in batch_mods:
            emb = self.fp(**batch_mods["fp"])
            out["fp"] = F.normalize(self.proj_fp(emb), dim=-1)
        if self.psmiles is not None and "psmiles" in batch_mods:
            emb = self.psmiles(**batch_mods["psmiles"])
            out["psmiles"] = F.normalize(self.proj_psmiles(emb), dim=-1)
        return out


@dataclass
class LoadedMultimodalModel:
    """Bundle storing the instantiated model and the device it lives on."""

    model: MultimodalContrastiveModel
    device: torch.device

    def encode(self, batch_mods: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.model.to(self.device)
        prepared: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, value in batch_mods.items():
            prepared[key] = {k: v.to(self.device) for k, v in value.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            return self.model.encode(prepared)
