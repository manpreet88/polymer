"""Reusable neural encoders leveraged by the Polymer RAG agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch_geometric is optional during documentation builds
    from torch_geometric.nn import GINEConv
    from torch_geometric.nn.models import SchNet as PyGSchNet
    from torch_geometric.nn import radius_graph
except Exception:  # pragma: no cover - handled gracefully at runtime
    GINEConv = None
    PyGSchNet = None
    radius_graph = None

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None


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
FP_HIDDEN = 256
FP_NUM_LAYERS = 4
FP_NUM_HEADS = 8
FP_FEEDFORWARD = 1024
PSMILES_MAX_LEN = 128
MULTIMODAL_EMB_DIM = 600
TEMPERATURE = 0.07


def match_edge_attr_to_index(edge_index: torch.Tensor, edge_attr: torch.Tensor, target_dim: int = 3) -> torch.Tensor:
    """Ensure edge attributes match the number of directed edges."""

    device = edge_attr.device if edge_attr is not None else (edge_index.device if edge_index is not None else "cpu")
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((0, target_dim), dtype=torch.float32, device=device)
    total_edges = edge_index.size(1)
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.zeros((total_edges, target_dim), dtype=torch.float32, device=device)
    if edge_attr.size(0) == total_edges:
        padded = edge_attr
    elif edge_attr.size(0) * 2 == total_edges:
        padded = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        repeats = (total_edges + edge_attr.size(0) - 1) // edge_attr.size(0)
        padded = edge_attr.repeat(repeats, 1)[:total_edges]
    feat_dim = padded.size(1)
    if feat_dim == target_dim:
        return padded
    if feat_dim < target_dim:
        pad = torch.zeros((total_edges, target_dim - feat_dim), dtype=torch.float32, device=padded.device)
        return torch.cat([padded, pad], dim=1)
    return padded[:, :target_dim]


class GineBlock(nn.Module):
    def __init__(self, node_dim: int) -> None:
        super().__init__()
        if GINEConv is None:
            raise RuntimeError("torch_geometric is required for the GINE encoder.")
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.conv = GINEConv(self.mlp)
        self.bn = nn.BatchNorm1d(node_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        return self.act(self.bn(h))


class GineEncoder(nn.Module):
    def __init__(self, node_emb_dim: int = NODE_EMB_DIM, edge_emb_dim: int = EDGE_EMB_DIM, num_layers: int = NUM_GNN_LAYERS) -> None:
        super().__init__()
        self.atom_emb = nn.Embedding(num_embeddings=MASK_ATOM_ID + 1, embedding_dim=node_emb_dim)
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
        self.edge_proj = (
            nn.Linear(edge_emb_dim, node_emb_dim) if edge_emb_dim != node_emb_dim else nn.Identity()
        )
        self.layers = nn.ModuleList([GineBlock(node_emb_dim) for _ in range(num_layers)])
        self.pool_proj = nn.Linear(node_emb_dim, node_emb_dim)

    def forward(
        self,
        z: torch.Tensor,
        chirality: Optional[torch.Tensor],
        formal_charge: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = self.atom_emb.weight.device
        z = z.to(device)
        if chirality is None:
            chirality = torch.zeros_like(z, dtype=torch.float32, device=device)
        else:
            chirality = chirality.to(device)
        if formal_charge is None:
            formal_charge = torch.zeros_like(z, dtype=torch.float32, device=device)
        else:
            formal_charge = formal_charge.to(device)
        x = self.atom_emb(z) + self.node_attr_proj(torch.stack([chirality, formal_charge], dim=-1))
        if edge_attr is None:
            edge_tensor = torch.zeros((0, 3), dtype=torch.float32, device=device)
        else:
            edge_tensor = edge_attr.to(device)
        edge_attr = match_edge_attr_to_index(edge_index.to(device), edge_tensor)
        edge_attr = self.edge_proj(edge_attr)
        h = x
        for layer in self.layers:
            h = layer(h, edge_index.to(device), edge_attr)
        if batch is None:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            pooled = torch.zeros((num_graphs, h.size(1)), device=device)
            for idx in range(num_graphs):
                mask = batch == idx
                if mask.any():
                    pooled[idx] = h[mask].mean(dim=0)
        return self.pool_proj(pooled)


class NodeSchNetWrapper(nn.Module):
    def __init__(
        self,
        hidden_channels: int = SCHNET_HIDDEN,
        num_interactions: int = SCHNET_NUM_INTERACTIONS,
        num_gaussians: int = SCHNET_NUM_GAUSSIANS,
        cutoff: float = SCHNET_CUTOFF,
        max_num_neighbors: int = SCHNET_MAX_NEIGHBORS,
    ) -> None:
        super().__init__()
        if PyGSchNet is None:
            raise RuntimeError("torch_geometric is required for the SchNet encoder.")
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
        device = self.pool_proj.weight.device
        z = z.to(device)
        pos = pos.to(device)
        if radius_graph is None:
            raise RuntimeError("torch_geometric is required for the SchNet encoder.")
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=device)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        node_embeddings = self.schnet(z=z, pos=pos, batch=batch, edge_index=edge_index)
        if isinstance(node_embeddings, torch.Tensor) and node_embeddings.dim() == 2:
            node_h = node_embeddings
        else:
            node_h = node_embeddings[0]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled = torch.zeros((num_graphs, node_h.size(1)), device=device)
        for idx in range(num_graphs):
            mask = batch == idx
            if mask.any():
                pooled[idx] = node_h[mask].mean(dim=0)
        return self.pool_proj(pooled)


class FingerprintEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int = FP_LENGTH,
        hidden_dim: int = FP_HIDDEN,
        num_layers: int = FP_NUM_LAYERS,
        nhead: int = FP_NUM_HEADS,
        dim_feedforward: int = FP_FEEDFORWARD,
        vocab_size: int = VOCAB_SIZE_FP,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.token_emb.weight.device
        input_ids = input_ids.to(device)
        batch, length = input_ids.shape
        positions = torch.arange(length, device=device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        if attention_mask is not None:
            attn_mask = ~attention_mask.to(device)
            x = self.encoder(x, src_key_padding_mask=attn_mask)
        else:
            x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.pool_proj(pooled)


class PSMILESDebertaEncoder(nn.Module):
    def __init__(self, model_dir_or_name: Optional[str] = None) -> None:
        super().__init__()
        if AutoModel is None or AutoTokenizer is None:
            raise RuntimeError("transformers is required for the PSMILES encoder.")
        model_name = model_dir_or_name or "microsoft/deberta-v3-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        hidden_dim = self.model.config.hidden_size
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def tokenize(self, texts, max_length: int = PSMILES_MAX_LEN) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.pool_proj.weight.device
        outputs = self.model(input_ids.to(device), attention_mask=attention_mask.to(device) if attention_mask is not None else None)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]
        pooled = hidden.mean(dim=1)
        return self.pool_proj(pooled)


class MultimodalContrastiveModel(nn.Module):
    def __init__(
        self,
        gine_encoder: Optional[GineEncoder] = None,
        schnet_encoder: Optional[NodeSchNetWrapper] = None,
        fingerprint_encoder: Optional[FingerprintEncoder] = None,
        psmiles_encoder: Optional[PSMILESDebertaEncoder] = None,
        emb_dim: int = MULTIMODAL_EMB_DIM,
    ) -> None:
        super().__init__()
        self.gine = gine_encoder
        self.schnet = schnet_encoder
        self.fp = fingerprint_encoder
        self.psmiles = psmiles_encoder
        self.proj_gine = nn.Linear(gine_encoder.pool_proj.out_features, emb_dim) if gine_encoder else None
        self.proj_schnet = nn.Linear(schnet_encoder.pool_proj.out_features, emb_dim) if schnet_encoder else None
        self.proj_fp = nn.Linear(fingerprint_encoder.pool_proj.out_features, emb_dim) if fingerprint_encoder else None
        self.proj_psmiles = nn.Linear(psmiles_encoder.pool_proj.out_features, emb_dim) if psmiles_encoder else None
        self.temperature = TEMPERATURE

    def encode(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        embeddings: Dict[str, torch.Tensor] = {}
        if self.gine is not None and "gine" in batch:
            emb = self.gine(**batch["gine"])
            embeddings["gine"] = F.normalize(self.proj_gine(emb), dim=-1)
        if self.schnet is not None and "schnet" in batch:
            emb = self.schnet(**batch["schnet"])
            embeddings["schnet"] = F.normalize(self.proj_schnet(emb), dim=-1)
        if self.fp is not None and "fp" in batch:
            emb = self.fp(**batch["fp"])
            embeddings["fingerprint"] = F.normalize(self.proj_fp(emb), dim=-1)
        if self.psmiles is not None and "psmiles" in batch:
            emb = self.psmiles(**batch["psmiles"])
            embeddings["psmiles"] = F.normalize(self.proj_psmiles(emb), dim=-1)
        if len(embeddings) >= 2:
            stacked = torch.stack(list(embeddings.values()), dim=0).mean(dim=0)
            embeddings["multimodal"] = F.normalize(stacked, dim=-1)
        return embeddings

    @classmethod
    def load_from_checkpoints(cls, checkpoint_dir: Path, device: Optional[torch.device] = None) -> "MultimodalContrastiveModel":
        device = device or torch.device("cpu")
        path = Path(checkpoint_dir) / "pytorch_model.bin"
        state = torch.load(path, map_location=device)
        model = cls()
        model.load_state_dict(state, strict=False)
        return model


__all__ = [
    "GineEncoder",
    "NodeSchNetWrapper",
    "FingerprintEncoder",
    "PSMILESDebertaEncoder",
    "MultimodalContrastiveModel",
]
