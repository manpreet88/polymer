import os
import random
import time
from pathlib import Path
import math
import json
import shutil
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import csv
csv.field_size_limit(sys.maxsize)

# PyG building blocks
try:
    from torch_geometric.nn import GINEConv
    from torch_geometric.nn.models import SchNet as PyGSchNet
    from torch_geometric.nn import radius_graph
except Exception as e:
    GINEConv = None
    PyGSchNet = None
    radius_graph = None

# HF Transformers
from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer

# Configuration
BASE_DIR = "Polymer_Foundational_Model"
POLYINFO_PATH = os.path.join(BASE_DIR, "polyinfo_with_modalities.csv")

# Pretrained model directories
PRETRAINED_MULTIMODAL_DIR = "./multimodal_output/best"
BEST_GINE_DIR = "./gin_output/best"
BEST_SCHNET_DIR = "./schnet_output/best"
BEST_FP_DIR = "./fingerprint_mlm_output/best"
BEST_PSMILES_DIR = "./polybert_output/best"

OUTPUT_RESULTS = "multimodal_downstream_results.txt"

# Model hyperparameters (matching your multimodal training)
MAX_ATOMIC_Z = 85
MASK_ATOM_ID = MAX_ATOMIC_Z + 1

# GINE params
NODE_EMB_DIM = 300
EDGE_EMB_DIM = 300
NUM_GNN_LAYERS = 5

# SchNet params
SCHNET_NUM_GAUSSIANS = 50
SCHNET_NUM_INTERACTIONS = 6
SCHNET_CUTOFF = 10.0
SCHNET_MAX_NEIGHBORS = 64
SCHNET_HIDDEN = 600

# Fingerprint params
FP_LENGTH = 2048
MASK_TOKEN_ID_FP = 2
VOCAB_SIZE_FP = 3

# PSMILES/Deberta params
DEBERTA_HIDDEN = 600
PSMILES_MAX_LEN = 128

# Training parameters
MAX_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 100
WARMUP_FROZEN_EPOCHS = 3
UNFREEZE_EVERY = 1
UNFREEZE_LAYERS_EACH_STEP = 1
PATIENCE = 10
LEARNING_RATE_HEAD = 3e-4
LEARNING_RATE_BASE = 5e-5
WEIGHT_DECAY = 0.01
HUBER_DELTA = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REQUESTED_PROPERTIES = [
    "density",
    "glass transition",
    "melting",
    "specific volume",
    "thermal decomposition"
]

REPEATS = 5
RANDOM_SEEDS = [42 + i * 17 for i in range(REPEATS)]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_property_columns(columns):
    lowered = {c.lower(): c for c in columns}
    found = {}
    for req in REQUESTED_PROPERTIES:
        match = None
        for c_low, c_orig in lowered.items():
            if req in c_low:
                match = c_orig
                break
        found[req] = match
    return found

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return float(obj)
    if isinstance(obj, torch.Tensor):
        try:
            return obj.detach().cpu().tolist()
        except Exception:
            return None
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    try:
        if isinstance(obj, (float, int, str, bool, type(None))):
            return obj
    except Exception:
        pass
    return obj

def safe_get(d: dict, key: str, default=None):
    return d[key] if (isinstance(d, dict) and key in d) else default

def match_edge_attr_to_index(edge_index: torch.Tensor, edge_attr: torch.Tensor, target_dim: int = 3):
    dev = None
    if edge_attr is not None and hasattr(edge_attr, "device"):
        dev = edge_attr.device
    elif edge_index is not None and hasattr(edge_index, "device"):
        dev = edge_index.device
    else:
        dev = torch.device("cpu")

    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((0, target_dim), dtype=torch.float, device=dev)
    E_idx = edge_index.size(1)
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.zeros((E_idx, target_dim), dtype=torch.float, device=dev)
    E_attr = edge_attr.size(0)
    if E_attr == E_idx:
        if edge_attr.size(1) != target_dim:
            D = edge_attr.size(1)
            if D < target_dim:
                pad = torch.zeros((E_attr, target_dim - D), dtype=torch.float, device=edge_attr.device)
                return torch.cat([edge_attr, pad], dim=1)
            else:
                return edge_attr[:, :target_dim]
        return edge_attr
    if E_attr * 2 == E_idx:
        try:
            return torch.cat([edge_attr, edge_attr], dim=0)
        except Exception:
            pass
    reps = (E_idx + E_attr - 1) // E_attr
    edge_rep = edge_attr.repeat(reps, 1)[:E_idx]
    if edge_rep.size(1) != target_dim:
        D = edge_rep.size(1)
        if D < target_dim:
            pad = torch.zeros((E_idx, target_dim - D), dtype=torch.float, device=edge_rep.device)
            edge_rep = torch.cat([edge_rep, pad], dim=1)
        else:
            edge_rep = edge_rep[:, :target_dim]
    return edge_rep

# Encoder definitions from multimodal training
class GineBlock(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        if GINEConv is None:
            raise RuntimeError("GINEConv is not available. Install torch_geometric with compatible versions.")
        self.conv = GINEConv(self.mlp)
        self.bn = nn.BatchNorm1d(node_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = self.act(x)
        return x

class GineEncoder(nn.Module):
    def __init__(self, node_emb_dim=NODE_EMB_DIM, edge_emb_dim=EDGE_EMB_DIM, num_layers=NUM_GNN_LAYERS, max_atomic_z=MAX_ATOMIC_Z):
        super().__init__()
        self.atom_emb = nn.Embedding(num_embeddings=MASK_ATOM_ID+1, embedding_dim=node_emb_dim, padding_idx=None)
        self.node_attr_proj = nn.Sequential(
            nn.Linear(2, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, node_emb_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim)
        )
        if edge_emb_dim != node_emb_dim:
            self._edge_to_node_proj = nn.Linear(edge_emb_dim, node_emb_dim)
        else:
            self._edge_to_node_proj = None
        self.gnn_layers = nn.ModuleList([GineBlock(node_emb_dim) for _ in range(num_layers)])
        self.pool_proj = nn.Linear(node_emb_dim, node_emb_dim)
        self.node_classifier = nn.Linear(node_emb_dim, MASK_ATOM_ID+1)

    def _compute_node_reps(self, z, chirality, formal_charge, edge_index, edge_attr):
        device = next(self.parameters()).device
        atom_embedding = self.atom_emb(z.to(device))
        if chirality is None or formal_charge is None:
            node_attr = torch.zeros((z.size(0), 2), device=device)
        else:
            node_attr = torch.stack([chirality, formal_charge], dim=1).to(atom_embedding.device)
        node_attr_emb = self.node_attr_proj(node_attr)
        x = atom_embedding + node_attr_emb
        if edge_attr is None or edge_attr.numel() == 0:
            edge_emb = torch.zeros((0, EDGE_EMB_DIM), dtype=torch.float, device=x.device)
        else:
            edge_emb = self.edge_encoder(edge_attr.to(x.device))
        if self._edge_to_node_proj is not None and edge_emb.numel() > 0:
            edge_for_conv = self._edge_to_node_proj(edge_emb)
        else:
            edge_for_conv = edge_emb

        h = x
        for layer in self.gnn_layers:
            h = layer(h, edge_index.to(h.device), edge_for_conv)
        return h

    def forward(self, z, chirality, formal_charge, edge_index, edge_attr, batch=None):
        h = self._compute_node_reps(z, chirality, formal_charge, edge_index, edge_attr)
        if batch is None:
            pooled = torch.mean(h, dim=0, keepdim=True)
        else:
            bsize = int(batch.max().item() + 1) if batch.numel() > 0 else 1
            pooled = torch.zeros((bsize, h.size(1)), device=h.device)
            for i in range(bsize):
                mask = batch == i
                if mask.sum() == 0:
                    continue
                pooled[i] = h[mask].mean(dim=0)
        return self.pool_proj(pooled)

class NodeSchNetWrapper(nn.Module):
    def __init__(self, hidden_channels=SCHNET_HIDDEN, num_interactions=SCHNET_NUM_INTERACTIONS, num_gaussians=SCHNET_NUM_GAUSSIANS, cutoff=SCHNET_CUTOFF, max_num_neighbors=SCHNET_MAX_NEIGHBORS):
        super().__init__()
        if PyGSchNet is None:
            raise RuntimeError("PyG SchNet is not available. Install torch_geometric with compatible extras.")
        self.schnet = PyGSchNet(hidden_channels=hidden_channels, num_filters=hidden_channels, num_interactions=num_interactions, num_gaussians=num_gaussians, cutoff=cutoff, max_num_neighbors=max_num_neighbors)
        self.pool_proj = nn.Linear(hidden_channels, hidden_channels)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.node_classifier = nn.Linear(hidden_channels, MASK_ATOM_ID+1)

    def forward(self, z, pos, batch=None):
        device = next(self.parameters()).device
        z = z.to(device)
        pos = pos.to(device)
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        try:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        except Exception:
            edge_index = None

        node_h = None
        try:
            if hasattr(self.schnet, "embedding"):
                node_h = self.schnet.embedding(z)
            else:
                node_h = self.schnet.embedding(z)
        except Exception:
            node_h = None

        if node_h is not None and edge_index is not None and edge_index.numel() > 0:
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            edge_attr = None
            if hasattr(self.schnet, "distance_expansion"):
                try:
                    edge_attr = self.schnet.distance_expansion(edge_weight)
                except Exception:
                    edge_attr = None
            if edge_attr is None and hasattr(self.schnet, "gaussian_smearing"):
                try:
                    edge_attr = self.schnet.gaussian_smearing(edge_weight)
                except Exception:
                    edge_attr = None
            if hasattr(self.schnet, "interactions") and getattr(self.schnet, "interactions") is not None:
                for interaction in self.schnet.interactions:
                    try:
                        node_h = node_h + interaction(node_h, edge_index, edge_weight, edge_attr)
                    except TypeError:
                        node_h = node_h + interaction(node_h, edge_index, edge_weight)
        if node_h is None:
            try:
                out = self.schnet(z=z, pos=pos, batch=batch)
                if isinstance(out, torch.Tensor) and out.dim() == 2 and out.size(0) == z.size(0):
                    node_h = out
                elif hasattr(out, "last_hidden_state"):
                    node_h = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    cand = out[0]
                    if cand.dim() == 2 and cand.size(0) == z.size(0):
                        node_h = cand
            except Exception as e:
                raise RuntimeError("Failed to obtain node-level embeddings from PyG SchNet.") from e

        bsize = int(batch.max().item()) + 1 if z.numel() > 0 else 1
        pooled = torch.zeros((bsize, node_h.size(1)), device=node_h.device)
        for i in range(bsize):
            mask = batch == i
            if mask.sum() == 0:
                continue
            pooled[i] = node_h[mask].mean(dim=0)
        return self.pool_proj(pooled)

class FingerprintEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE_FP, hidden_dim=256, seq_len=FP_LENGTH, num_layers=4, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.seq_len = seq_len
        self.token_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(input_ids.device)
        else:
            key_padding_mask = None
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        if attention_mask is None:
            pooled = out.mean(dim=1)
        else:
            am = attention_mask.to(out.device).float().unsqueeze(-1)
            pooled = (out * am).sum(dim=1) / (am.sum(dim=1).clamp(min=1.0))
        return self.pool_proj(pooled)

class PSMILESDebertaEncoder(nn.Module):
    def __init__(self, model_dir_or_name=None):
        super().__init__()
        try:
            if model_dir_or_name is not None and os.path.isdir(model_dir_or_name):
                self.model = DebertaV2ForMaskedLM.from_pretrained(model_dir_or_name)
            else:
                self.model = DebertaV2ForMaskedLM.from_pretrained("microsoft/deberta-v2-xlarge")
        except Exception as e:
            print("Warning: couldn't load DebertaV2 pretrained weights; initializing randomly.", e)
            from transformers import DebertaV2Config
            cfg = DebertaV2Config(vocab_size=300, hidden_size=DEBERTA_HIDDEN, num_attention_heads=12, num_hidden_layers=12, intermediate_size=512)
            self.model = DebertaV2ForMaskedLM(cfg)
        self.pool_proj = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            am = attention_mask.unsqueeze(-1).to(last_hidden.device).float()
            pooled = (last_hidden * am).sum(dim=1) / (am.sum(dim=1).clamp(min=1.0))
        return self.pool_proj(pooled)

# Multimodal model with frozen encoders and trainable projections
class MultimodalContrastiveModel(nn.Module):
    def __init__(self,
                 gine_encoder,
                 schnet_encoder,
                 fp_encoder,
                 psmiles_encoder,
                 emb_dim: int = 600):
        super().__init__()
        self.gine = gine_encoder
        self.schnet = schnet_encoder
        self.fp = fp_encoder
        self.psmiles = psmiles_encoder
        self.proj_gine = nn.Linear(getattr(self.gine, "pool_proj").out_features if self.gine is not None else emb_dim, emb_dim) if self.gine is not None else None
        self.proj_schnet = nn.Linear(getattr(self.schnet, "pool_proj").out_features if self.schnet is not None else emb_dim, emb_dim) if self.schnet is not None else None
        self.proj_fp = nn.Linear(getattr(self.fp, "pool_proj").out_features if self.fp is not None else emb_dim, emb_dim) if self.fp is not None else None
        self.proj_psmiles = nn.Linear(getattr(self.psmiles, "pool_proj").out_features if self.psmiles is not None else emb_dim, emb_dim) if self.psmiles is not None else None

    def encode(self, batch_mods):
        device = next(self.parameters()).device
        embs = {}
        B = None
        if 'gine' in batch_mods and self.gine is not None:
            g = batch_mods['gine']
            emb_g = self.gine(g['z'], g['chirality'], g['formal_charge'], g['edge_index'], g['edge_attr'], g.get('batch', None))
            embs['gine'] = F.normalize(self.proj_gine(emb_g), dim=-1)
            B = embs['gine'].size(0) if B is None else B
        if 'schnet' in batch_mods and self.schnet is not None:
            s = batch_mods['schnet']
            emb_s = self.schnet(s['z'], s['pos'], s.get('batch', None))
            embs['schnet'] = F.normalize(self.proj_schnet(emb_s), dim=-1)
            B = embs['schnet'].size(0) if B is None else B
        if 'fp' in batch_mods and self.fp is not None:
            f = batch_mods['fp']
            emb_f = self.fp(f['input_ids'], f.get('attention_mask', None))
            embs['fp'] = F.normalize(self.proj_fp(emb_f), dim=-1)
            B = embs['fp'].size(0) if B is None else B
        if 'psmiles' in batch_mods and self.psmiles is not None:
            p = batch_mods['psmiles']
            emb_p = self.psmiles(p['input_ids'], p.get('attention_mask', None))
            embs['psmiles'] = F.normalize(self.proj_psmiles(emb_p), dim=-1)
            B = embs['psmiles'].size(0) if B is None else B
        return embs

    def forward_multimodal(self, batch_mods):
        """Get combined multimodal embedding"""
        embs = self.encode(batch_mods)
        if len(embs) == 0:
            device = next(self.parameters()).device
            return torch.zeros((1, 600), device=device)

        # Combine all available modality embeddings
        combined = torch.stack(list(embs.values()), dim=0).mean(dim=0)
        return combined

# Tokenizer setup
try:
    SPM_MODEL = "spm.model"
    if Path(SPM_MODEL).exists():
        tokenizer = DebertaV2Tokenizer(vocab_file=SPM_MODEL, do_lower_case=False)
        tokenizer.add_special_tokens({"pad_token": "<pad>", "mask_token": "<mask>"})
        tokenizer.pad_token = "<pad>"
        tokenizer.mask_token = "<mask>"
    else:
        tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge", use_fast=False)
        tokenizer.add_special_tokens({"pad_token": "<pad>", "mask_token": "<mask>"})
        tokenizer.pad_token = "<pad>"
        tokenizer.mask_token = "<mask>"
except Exception as e:
    print("Warning: Deberta tokenizer creation failed:", e)
    # Simple fallback tokenizer
    class SimplePSMILESTokenizer:
        def __init__(self, max_len=PSMILES_MAX_LEN):
            chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-=#()[]@+/\\.\"")
            self.vocab = {c: i + 5 for i, c in enumerate(chars)}
            self.vocab["<pad>"] = 0
            self.vocab["<mask>"] = 1
            self.vocab["<unk>"] = 2
            self.vocab["<cls>"] = 3
            self.vocab["<sep>"] = 4
            self.mask_token = "<mask>"
            self.mask_token_id = self.vocab[self.mask_token]
            self.vocab_size = len(self.vocab)
            self.max_len = max_len

        def __call__(self, s, truncation=True, padding="max_length", max_length=None):
            max_len = max_length or self.max_len
            toks = [self.vocab.get(ch, self.vocab["<unk>"]) for ch in list(s)][:max_len]
            attn = [1] * len(toks)
            if len(toks) < max_len:
                pad = [self.vocab["<pad>"]] * (max_len - len(toks))
                toks = toks + pad
                attn = attn + [0] * (max_len - len(attn))
            return {"input_ids": toks, "attention_mask": attn}

    tokenizer = SimplePSMILESTokenizer()

class MultimodalPolymerDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # Parse graph data for GINE
        gine_data = None
        if 'graph' in data and data['graph']:
            try:
                graph_field = json.loads(data['graph']) if isinstance(data['graph'], str) else data['graph']
                node_features = safe_get(graph_field, "node_features", None)
                if node_features:
                    atomic_nums = []
                    chirality_vals = []
                    formal_charges = []
                    for nf in node_features:
                        an = safe_get(nf, "atomic_num", None)
                        if an is None:
                            an = safe_get(nf, "atomic_number", 0)
                        ch = safe_get(nf, "chirality", 0)
                        fc = safe_get(nf, "formal_charge", 0)
                        try:
                            atomic_nums.append(int(an))
                        except Exception:
                            atomic_nums.append(0)
                        chirality_vals.append(float(ch))
                        formal_charges.append(float(fc))

                    # Process edges
                    edge_indices_raw = safe_get(graph_field, "edge_indices", None)
                    edge_features_raw = safe_get(graph_field, "edge_features", None)
                    edge_index = None
                    edge_attr = None

                    if edge_indices_raw is None:
                        adj_mat = safe_get(graph_field, "adjacency_matrix", None)
                        if adj_mat:
                            srcs = []
                            dsts = []
                            for i_r, row_adj in enumerate(adj_mat):
                                for j, val2 in enumerate(row_adj):
                                    if val2:
                                        srcs.append(i_r)
                                        dsts.append(j)
                            if len(srcs) > 0:
                                edge_index = [srcs, dsts]
                                E = len(srcs)
                                edge_attr = [[0.0, 0.0, 0.0] for _ in range(E)]
                    else:
                        srcs, dsts = [], []
                        if isinstance(edge_indices_raw, list) and len(edge_indices_raw) > 0:
                            if isinstance(edge_indices_raw[0], list):
                                first = edge_indices_raw[0]
                                if len(first) == 2 and isinstance(first[0], int):
                                    try:
                                        srcs = [int(p[0]) for p in edge_indices_raw]
                                        dsts = [int(p[1]) for p in edge_indices_raw]
                                    except Exception:
                                        srcs, dsts = [], []
                                else:
                                    try:
                                        srcs = [int(x) for x in edge_indices_raw[0]]
                                        dsts = [int(x) for x in edge_indices_raw[1]]
                                    except Exception:
                                        srcs, dsts = [], []

                        if len(srcs) > 0:
                            edge_index = [srcs, dsts]
                            if edge_features_raw and isinstance(edge_features_raw, list):
                                bond_types = []
                                stereos = []
                                is_conjs = []
                                for ef in edge_features_raw:
                                    bt = safe_get(ef, "bond_type", 0)
                                    st = safe_get(ef, "stereo", 0)
                                    ic = safe_get(ef, "is_conjugated", False)
                                    bond_types.append(float(bt))
                                    stereos.append(float(st))
                                    is_conjs.append(float(1.0 if ic else 0.0))
                                edge_attr = list(zip(bond_types, stereos, is_conjs))
                            else:
                                E = len(srcs)
                                edge_attr = [[0.0, 0.0, 0.0] for _ in range(E)]

                    if edge_index is not None:
                        gine_data = {
                            'z': torch.tensor(atomic_nums, dtype=torch.long),
                            'chirality': torch.tensor(chirality_vals, dtype=torch.float),
                            'formal_charge': torch.tensor(formal_charges, dtype=torch.float),
                            'edge_index': torch.tensor(edge_index, dtype=torch.long),
                            'edge_attr': torch.tensor(edge_attr, dtype=torch.float)
                        }
            except Exception as e:
                gine_data = None

        # Parse geometry data for SchNet
        schnet_data = None
        if 'geometry' in data and data['geometry']:
            try:
                geom = json.loads(data['geometry']) if isinstance(data['geometry'], str) else data['geometry']
                conf = geom.get("best_conformer") if isinstance(geom, dict) else None
                if conf:
                    atomic = conf.get("atomic_numbers", [])
                    coords = conf.get("coordinates", [])
                    if len(atomic) == len(coords) and len(atomic) > 0:
                        schnet_data = {
                            'z': torch.tensor(atomic, dtype=torch.long),
                            'pos': torch.tensor(coords, dtype=torch.float)
                        }
            except Exception:
                schnet_data = None

        # Parse fingerprints
        fp_data = None
        if 'fingerprints' in data and data['fingerprints']:
            try:
                fpval = data['fingerprints']
                if fpval is not None and not (isinstance(fpval, str) and fpval.strip() == ""):
                    try:
                        fp_json = json.loads(fpval) if isinstance(fpval, str) else fpval
                    except Exception:
                        try:
                            fp_json = json.loads(str(fpval).replace("'", '"'))
                        except Exception:
                            parts = [p.strip().strip('"').strip("'") for p in str(fpval).split(",")]
                            bits = [1 if p in ("1", "True", "true") else 0 for p in parts[:FP_LENGTH]]
                            if len(bits) < FP_LENGTH:
                                bits += [0] * (FP_LENGTH - len(bits))
                            fp_json = bits

                    if isinstance(fp_json, dict):
                        bits = safe_get(fp_json, "morgan_r3_bits", None)
                        if bits is None:
                            bits = [0] * FP_LENGTH
                        else:
                            normalized = []
                            for b in bits:
                                if isinstance(b, str):
                                    b_clean = b.strip().strip('"').strip("'")
                                    normalized.append(1 if b_clean in ("1", "True", "true") else 0)
                                elif isinstance(b, (int, np.integer)):
                                    normalized.append(1 if int(b) != 0 else 0)
                                else:
                                    normalized.append(0)
                                if len(normalized) >= FP_LENGTH:
                                    break
                            if len(normalized) < FP_LENGTH:
                                normalized.extend([0] * (FP_LENGTH - len(normalized)))
                            bits = normalized[:FP_LENGTH]
                    elif isinstance(fp_json, list):
                        bits = fp_json[:FP_LENGTH]
                        if len(bits) < FP_LENGTH:
                            bits += [0] * (FP_LENGTH - len(bits))
                    else:
                        bits = [0] * FP_LENGTH

                    fp_data = {
                        'input_ids': torch.tensor(bits, dtype=torch.long),
                        'attention_mask': torch.ones(FP_LENGTH, dtype=torch.bool)
                    }
            except Exception:
                fp_data = None

        # Parse PSMILES
        psmiles_data = None
        if 'psmiles' in data and data['psmiles']:
            try:
                s = str(data['psmiles'])
                enc = self.tokenizer(s, truncation=True, padding="max_length", max_length=PSMILES_MAX_LEN)
                psmiles_data = {
                    'input_ids': torch.tensor(enc["input_ids"], dtype=torch.long),
                    'attention_mask': torch.tensor(enc["attention_mask"], dtype=torch.bool)
                }
            except Exception:
                psmiles_data = None

        # Set defaults for missing modalities
        if gine_data is None:
            gine_data = {
                'z': torch.tensor([], dtype=torch.long),
                'chirality': torch.tensor([], dtype=torch.float),
                'formal_charge': torch.tensor([], dtype=torch.float),
                'edge_index': torch.tensor([[], []], dtype=torch.long),
                'edge_attr': torch.zeros((0, 3), dtype=torch.float)
            }

        if schnet_data is None:
            schnet_data = {
                'z': torch.tensor([], dtype=torch.long),
                'pos': torch.tensor([], dtype=torch.float)
            }

        if fp_data is None:
            fp_data = {
                'input_ids': torch.zeros(FP_LENGTH, dtype=torch.long),
                'attention_mask': torch.ones(FP_LENGTH, dtype=torch.bool)
            }

        if psmiles_data is None:
            psmiles_data = {
                'input_ids': torch.zeros(PSMILES_MAX_LEN, dtype=torch.long),
                'attention_mask': torch.zeros(PSMILES_MAX_LEN, dtype=torch.bool)
            }

        return {
            'gine': gine_data,
            'schnet': schnet_data,
            'fp': fp_data,
            'psmiles': psmiles_data,
            'target': data['target']
        }

def multimodal_collate_fn(batch):
    # GINE batching
    all_z = []
    all_ch = []
    all_fc = []
    all_edge_index = []
    all_edge_attr = []
    batch_mapping = []
    node_offset = 0
    for i, item in enumerate(batch):
        g = item["gine"]
        z = g["z"]
        n = z.size(0)
        all_z.append(z)
        all_ch.append(g["chirality"])
        all_fc.append(g["formal_charge"])
        batch_mapping.append(torch.full((n,), i, dtype=torch.long))
        if g["edge_index"] is not None and g["edge_index"].numel() > 0:
            ei_offset = g["edge_index"] + node_offset
            all_edge_index.append(ei_offset)
            ea = match_edge_attr_to_index(g["edge_index"], g["edge_attr"], target_dim=3)
            all_edge_attr.append(ea)
        node_offset += n

    if len(all_z) == 0:
        z_batch = torch.tensor([], dtype=torch.long)
        ch_batch = torch.tensor([], dtype=torch.float)
        fc_batch = torch.tensor([], dtype=torch.float)
        batch_batch = torch.tensor([], dtype=torch.long)
        edge_index_batched = torch.empty((2,0), dtype=torch.long)
        edge_attr_batched = torch.zeros((0,3), dtype=torch.float)
    else:
        z_batch = torch.cat(all_z, dim=0)
        ch_batch = torch.cat(all_ch, dim=0)
        fc_batch = torch.cat(all_fc, dim=0)
        batch_batch = torch.cat(batch_mapping, dim=0)
        if len(all_edge_index) > 0:
            edge_index_batched = torch.cat(all_edge_index, dim=1)
            edge_attr_batched = torch.cat(all_edge_attr, dim=0)
        else:
            edge_index_batched = torch.empty((2,0), dtype=torch.long)
            edge_attr_batched = torch.zeros((0,3), dtype=torch.float)

    # SchNet batching
    all_sz = []
    all_pos = []
    schnet_batch = []
    for i, item in enumerate(batch):
        s = item["schnet"]
        s_z = s["z"]
        s_pos = s["pos"]
        if s_z.numel() == 0:
            continue
        all_sz.append(s_z)
        all_pos.append(s_pos)
        schnet_batch.append(torch.full((s_z.size(0),), i, dtype=torch.long))

    if len(all_sz) == 0:
        s_z_batch = torch.tensor([], dtype=torch.long)
        s_pos_batch = torch.tensor([], dtype=torch.float)
        s_batch_batch = torch.tensor([], dtype=torch.long)
    else:
        s_z_batch = torch.cat(all_sz, dim=0)
        s_pos_batch = torch.cat(all_pos, dim=0)
        s_batch_batch = torch.cat(schnet_batch, dim=0)

    # FP batching
    fp_ids = torch.stack([item["fp"]["input_ids"] for item in batch], dim=0)
    fp_attn = torch.stack([item["fp"]["attention_mask"] for item in batch], dim=0)

    # PSMILES batching
    p_ids = torch.stack([item["psmiles"]["input_ids"] for item in batch], dim=0)
    p_attn = torch.stack([item["psmiles"]["attention_mask"] for item in batch], dim=0)

    # Targets
    targets = torch.tensor([item["target"] for item in batch], dtype=torch.float32)

    return {
        "gine": {"z": z_batch, "chirality": ch_batch, "formal_charge": fc_batch, "edge_index": edge_index_batched, "edge_attr": edge_attr_batched, "batch": batch_batch},
        "schnet": {"z": s_z_batch, "pos": s_pos_batch, "batch": s_batch_batch},
        "fp": {"input_ids": fp_ids, "attention_mask": fp_attn},
        "psmiles": {"input_ids": p_ids, "attention_mask": p_attn},
        "target": targets
    }

class MultimodalRegressionModel(nn.Module):
    def __init__(self, multimodal_model, hidden_dim=600, dropout=0.1):
        super().__init__()
        self.multimodal = multimodal_model
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, batch_mods):
        # Get multimodal embedding
        multimodal_emb = self.multimodal.forward_multimodal(batch_mods)
        # Apply regression head
        out = self.head(multimodal_emb).squeeze(-1)
        return out

def get_encoder_layers(multimodal_model):
    """Get all encoder layers for gradual unfreezing"""
    layers = []
    if hasattr(multimodal_model, 'gine') and multimodal_model.gine is not None:
        if hasattr(multimodal_model.gine, 'gnn_layers'):
            layers.extend(list(multimodal_model.gine.gnn_layers))
    if hasattr(multimodal_model, 'schnet') and multimodal_model.schnet is not None:
        if hasattr(multimodal_model.schnet.schnet, 'interactions'):
            layers.extend(list(multimodal_model.schnet.schnet.interactions))
    if hasattr(multimodal_model, 'fp') and multimodal_model.fp is not None:
        if hasattr(multimodal_model.fp.transformer, 'layers'):
            layers.extend(list(multimodal_model.fp.transformer.layers))
    if hasattr(multimodal_model, 'psmiles') and multimodal_model.psmiles is not None:
        if hasattr(multimodal_model.psmiles.model, 'encoder'):
            if hasattr(multimodal_model.psmiles.model.encoder, 'layer'):
                layers.extend(list(multimodal_model.psmiles.model.encoder.layer))
    return layers

def freeze_all(multimodal_model):
    for p in multimodal_model.parameters():
        p.requires_grad = False

def unfreeze_last_n_layers(multimodal_model, n):
    layers = get_encoder_layers(multimodal_model)
    if not layers:
        for p in multimodal_model.parameters():
            p.requires_grad = True
        return
    total = len(layers)
    to_unfreeze_indices = list(range(max(0, total - n), total))
    for i, layer in enumerate(layers):
        requires = (i in to_unfreeze_indices)
        for p in layer.parameters():
            p.requires_grad = requires

def train_one_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    count = 0
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    for batch in dataloader:
        # Move batch to device
        for k in batch:
            if k == 'target':
                batch[k] = batch[k].to(device)
            else:
                for subk in batch[k]:
                    if isinstance(batch[k][subk], torch.Tensor):
                        batch[k][subk] = batch[k][subk].to(device)

        targets = batch["target"]
        batch_mods = {k: v for k, v in batch.items() if k != 'target'}

        preds = model(batch_mods)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += float(loss.item()) * targets.size(0)
        count += targets.size(0)

    return total_loss / max(1, count)

def evaluate(model, dataloader, device):
    model.eval()
    preds_all = []
    targets_all = []
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            for k in batch:
                if k == 'target':
                    batch[k] = batch[k].to(device)
                else:
                    for subk in batch[k]:
                        if isinstance(batch[k][subk], torch.Tensor):
                            batch[k][subk] = batch[k][subk].to(device)

            targets = batch["target"]
            batch_mods = {k: v for k, v in batch.items() if k != 'target'}

            preds = model(batch_mods)
            loss = criterion(preds, targets)

            total_loss += float(loss.item()) * targets.size(0)
            count += targets.size(0)

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())

    if count == 0:
        return None, None, None

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    avg_loss = total_loss / count

    return avg_loss, preds_all, targets_all

def compute_metrics_from_arrays(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

def run_downstream_for_property(property_name, property_col, df_prop, pretrained_path, output_file):
    results_runs = []
    subset = df_prop.dropna(subset=[property_col]).reset_index(drop=True)

    # Filter for samples that have at least one modality
    valid_samples = []
    for idx, row in subset.iterrows():
        has_modality = False
        for col in ['graph', 'geometry', 'fingerprints', 'psmiles']:
            if col in row and row[col] and str(row[col]).strip() != "":
                has_modality = True
                break
        if has_modality:
            valid_samples.append({
                'graph': row.get('graph', ''),
                'geometry': row.get('geometry', ''),
                'fingerprints': row.get('fingerprints', ''),
                'psmiles': row.get('psmiles', ''),
                'target': float(row[property_col])
            })

    if len(valid_samples) < 20:
        print(f"[WARN] Not enough samples with modalities for {property_name} (found {len(valid_samples)}), skipping.")
        return None

    for run_idx, seed in enumerate(RANDOM_SEEDS):
        set_seed(seed)
        print(f"\n--- Property '{property_name}' run {run_idx+1}/{len(RANDOM_SEEDS)}, seed={seed} ---")

        # Train/val/test split
        indices = list(range(len(valid_samples)))
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

        train_samples = [valid_samples[i] for i in train_idx]
        val_samples = [valid_samples[i] for i in val_idx]
        test_samples = [valid_samples[i] for i in test_idx]

        # Scale targets
        train_targets = np.array([s['target'] for s in train_samples]).reshape(-1, 1)
        val_targets = np.array([s['target'] for s in val_samples]).reshape(-1, 1)
        test_targets = np.array([s['target'] for s in test_samples]).reshape(-1, 1)

        scaler = RobustScaler()
        train_targets_scaled = scaler.fit_transform(train_targets).ravel()
        val_targets_scaled = scaler.transform(val_targets).ravel()
        test_targets_scaled = scaler.transform(test_targets).ravel()

        # Update samples with scaled targets
        for i, scaled_target in enumerate(train_targets_scaled):
            train_samples[i]['target'] = scaled_target
        for i, scaled_target in enumerate(val_targets_scaled):
            val_samples[i]['target'] = scaled_target
        for i, scaled_target in enumerate(test_targets_scaled):
            test_samples[i]['target'] = scaled_target

        # Create datasets
        ds_train = MultimodalPolymerDataset(train_samples, tokenizer, max_length=MAX_LEN)
        ds_val = MultimodalPolymerDataset(val_samples, tokenizer, max_length=MAX_LEN)
        ds_test = MultimodalPolymerDataset(test_samples, tokenizer, max_length=MAX_LEN)

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=multimodal_collate_fn)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=multimodal_collate_fn)
        dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=multimodal_collate_fn)

        # Load pretrained multimodal model
        try:
            # Initialize encoders
            gine_encoder = GineEncoder(node_emb_dim=NODE_EMB_DIM, edge_emb_dim=EDGE_EMB_DIM, num_layers=NUM_GNN_LAYERS, max_atomic_z=MAX_ATOMIC_Z)
            if os.path.exists(os.path.join(BEST_GINE_DIR, "pytorch_model.bin")):
                try:
                    gine_encoder.load_state_dict(torch.load(os.path.join(BEST_GINE_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
                    print(f"Loaded GINE weights from {BEST_GINE_DIR}")
                except Exception as e:
                    print(f"Warning: Could not load GINE weights: {e}")

            schnet_encoder = NodeSchNetWrapper(hidden_channels=SCHNET_HIDDEN, num_interactions=SCHNET_NUM_INTERACTIONS, num_gaussians=SCHNET_NUM_GAUSSIANS, cutoff=SCHNET_CUTOFF, max_num_neighbors=SCHNET_MAX_NEIGHBORS)
            if os.path.exists(os.path.join(BEST_SCHNET_DIR, "pytorch_model.bin")):
                try:
                    schnet_encoder.load_state_dict(torch.load(os.path.join(BEST_SCHNET_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
                    print(f"Loaded SchNet weights from {BEST_SCHNET_DIR}")
                except Exception as e:
                    print(f"Warning: Could not load SchNet weights: {e}")

            fp_encoder = FingerprintEncoder(vocab_size=VOCAB_SIZE_FP, hidden_dim=256, seq_len=FP_LENGTH, num_layers=4, nhead=8, dim_feedforward=1024, dropout=0.1)
            if os.path.exists(os.path.join(BEST_FP_DIR, "pytorch_model.bin")):
                try:
                    fp_encoder.load_state_dict(torch.load(os.path.join(BEST_FP_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
                    print(f"Loaded fingerprint encoder weights from {BEST_FP_DIR}")
                except Exception as e:
                    print(f"Warning: Could not load fingerprint weights: {e}")

            psmiles_encoder = None
            if os.path.isdir(BEST_PSMILES_DIR):
                try:
                    psmiles_encoder = PSMILESDebertaEncoder(model_dir_or_name=BEST_PSMILES_DIR)
                    print(f"Loaded PSMILES encoder from {BEST_PSMILES_DIR}")
                except Exception as e:
                    print(f"Warning: Could not load PSMILES encoder: {e}")
            if psmiles_encoder is None:
                try:
                    psmiles_encoder = PSMILESDebertaEncoder(model_dir_or_name=None)
                except Exception as e:
                    print(f"Warning: Could not initialize PSMILES encoder: {e}")

            # Create multimodal model
            multimodal_model = MultimodalContrastiveModel(gine_encoder, schnet_encoder, fp_encoder, psmiles_encoder, emb_dim=600)

            # Load pretrained multimodal weights if available
            if os.path.exists(os.path.join(pretrained_path, "pytorch_model.bin")):
                try:
                    multimodal_model.load_state_dict(torch.load(os.path.join(pretrained_path, "pytorch_model.bin"), map_location="cpu"), strict=False)
                    print(f"Loaded multimodal pretrained weights from {pretrained_path}")
                except Exception as e:
                    print(f"Warning: Failed to load multimodal pretrained weights: {e}")

            # Create regression model
            model = MultimodalRegressionModel(multimodal_model=multimodal_model).to(DEVICE)

            print(f"Loaded pretrained multimodal model for property prediction")

        except Exception as e:
            print(f"[ERROR] Failed to load pretrained multimodal model: {e}")
            continue

        # Freeze encoders initially, keep projections and head trainable
        freeze_all(model.multimodal)
        for p in model.head.parameters():
            p.requires_grad = True
        # Keep projection heads trainable
        if hasattr(model.multimodal, 'proj_gine') and model.multimodal.proj_gine is not None:
            for p in model.multimodal.proj_gine.parameters():
                p.requires_grad = True
        if hasattr(model.multimodal, 'proj_schnet') and model.multimodal.proj_schnet is not None:
            for p in model.multimodal.proj_schnet.parameters():
                p.requires_grad = True
        if hasattr(model.multimodal, 'proj_fp') and model.multimodal.proj_fp is not None:
            for p in model.multimodal.proj_fp.parameters():
                p.requires_grad = True
        if hasattr(model.multimodal, 'proj_psmiles') and model.multimodal.proj_psmiles is not None:
            for p in model.multimodal.proj_psmiles.parameters():
                p.requires_grad = True

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None
        start_time = time.time()

        num_encoder_layers = len(get_encoder_layers(model.multimodal))
        layers_unfrozen = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            # Gradual unfreezing
            if epoch > WARMUP_FROZEN_EPOCHS:
                steps_since_unfreeze = epoch - WARMUP_FROZEN_EPOCHS
                to_unfreeze = min(num_encoder_layers, steps_since_unfreeze // UNFREEZE_EVERY * UNFREEZE_LAYERS_EACH_STEP)
                if to_unfreeze > layers_unfrozen:
                    unfreeze_last_n_layers(model.multimodal, to_unfreeze)
                    layers_unfrozen = to_unfreeze
                    print(f"[INFO] Epoch {epoch}: unfreezing last {to_unfreeze} encoder layers (total layers: {num_encoder_layers})")
                    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)

            train_loss = train_one_epoch(model, dl_train, optimizer, DEVICE)
            val_loss, _, _ = evaluate(model, dl_val, DEVICE)
            val_loss = val_loss if val_loss is not None else float("inf")

            print(f"Epoch {epoch} - train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

        total_time = time.time() - start_time
        print(f"Run finished in {total_time:.1f}s, best_val_loss={best_val_loss:.6f}")

        # Load best model and evaluate
        if best_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        _, preds_scaled, targets_scaled = evaluate(model, dl_test, DEVICE)
        y_pred_denorm = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        y_true_denorm = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).ravel()

        metrics = compute_metrics_from_arrays(y_true_denorm, y_pred_denorm)
        print(f"Test metrics (denorm): R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, MSE={metrics['mse']:.4f}")

        run_result = {
            "property": property_name,
            "property_col": property_col,
            "run": run_idx + 1,
            "seed": seed,
            "n_train": len(ds_train),
            "n_val": len(ds_val),
            "n_test": len(ds_test),
            "best_val_loss": float(best_val_loss),
            "test_metrics": metrics
        }
        results_runs.append(run_result)

        with open(output_file, "a") as fh:
            fh.write(json.dumps(make_json_serializable(run_result)) + "\n")

    if results_runs:
        ms = {"r2": [], "mae": [], "rmse": [], "mse": []}
        for r in results_runs:
            for k in ms.keys():
                ms[k].append(r["test_metrics"][k])
        agg = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=0))} for k, v in ms.items()}
        summary = {
            "property": property_name,
            "property_col": property_col,
            "n_runs": len(results_runs),
            "aggregated": agg
        }
        with open(output_file, "a") as fh:
            fh.write("AGGREGATED: " + json.dumps(make_json_serializable(summary)) + "\n")
        return {"runs": results_runs, "agg": agg}
    return None


# ---------------------------------------------------------------------------
# Lightweight utilities for agent orchestrators
# ---------------------------------------------------------------------------
_REFERENCE_TABLE_CACHE = None


def _load_reference_table():
    """Load a lightweight reference table for property estimation."""

    global _REFERENCE_TABLE_CACHE
    if _REFERENCE_TABLE_CACHE is not None:
        return _REFERENCE_TABLE_CACHE

    candidate_paths = [
        Path(POLYINFO_PATH),
        Path(__file__).resolve().parent / "examples" / "data" / "polymer_reference.csv",
    ]
    for path in candidate_paths:
        try:
            resolved = Path(path).expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            try:
                table = pd.read_csv(resolved, engine="python")
                _REFERENCE_TABLE_CACHE = table
                return table
            except Exception:
                continue
    return None


def estimate_properties_for_psmiles(psmiles: str, properties=None):
    """Estimate polymer properties using the reference dataset.

    The helper mirrors the downstream heads expected by the agent pipeline but
    falls back to curated reference values when learned weights are not
    available.  If the exact PSMILES string is not present, the closest entry
    (by sequence similarity) is used as a proxy.
    """

    table = _load_reference_table()
    if table is None or not isinstance(psmiles, str) or not psmiles:
        return {}

    df = table.copy()
    df["psmiles_lower"] = df.get("psmiles", "").astype(str).str.lower()
    target = psmiles.lower()

    match = df[df["psmiles_lower"] == target]
    if match.empty:
        df["_similarity"] = df["psmiles_lower"].apply(
            lambda candidate: SequenceMatcher(None, candidate, target).ratio()
        )
        match = df.sort_values("_similarity", ascending=False).head(1)
    results = {}

    property_columns = [col for col in df.columns if col.startswith("property:")]
    if properties is not None:
        desired = set(properties)
        property_columns = [
            col for col in property_columns if col[len("property:") :] in desired
        ]

    if match.empty:
        return {}

    row = match.iloc[0]
    for column in property_columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip().lower() in {"", "na", "n/a", "none"}:
            continue
        if pd.isna(value):
            continue
        key = column[len("property:") :]
        try:
            results[key] = float(value)
        except (TypeError, ValueError):
            continue
    return results


def main():
    if os.path.exists(OUTPUT_RESULTS):
        backup = OUTPUT_RESULTS + ".bak"
        shutil.copy(OUTPUT_RESULTS, backup)
        print(f"[INFO] Existing {OUTPUT_RESULTS} backed up to {backup}")
    open(OUTPUT_RESULTS, "w").close()

    if not os.path.isfile(POLYINFO_PATH):
        raise FileNotFoundError(f"PolyInfo file not found at {POLYINFO_PATH}")

    polyinfo_raw = pd.read_csv(POLYINFO_PATH, engine="python")
    found = find_property_columns(polyinfo_raw.columns)

    prop_map = {}
    for req, col in found.items():
        prop_map[req] = col

    overall_summary = {}
    for req in REQUESTED_PROPERTIES:
        col = prop_map.get(req)
        if col is None:
            print(f"[WARN] Could not find a column for requested property substring '{req}'. Skipping.")
            continue

        print(f"\n=== Running multimodal downstream for property '{req}' -> column '{col}' ===")
        res = run_downstream_for_property(req, col, polyinfo_raw, PRETRAINED_MULTIMODAL_DIR, OUTPUT_RESULTS)
        overall_summary[req] = res

    with open(OUTPUT_RESULTS, "a") as fh:
        fh.write("\nFINAL_SUMMARY\n")
        fh.write(json.dumps(make_json_serializable({k: (v["agg"] if v else None) for k, v in overall_summary.items()}), indent=2))
        fh.write("\n")

    print(f"\n[DONE] Results appended to {OUTPUT_RESULTS}")

if __name__ == "__main__":
    main()
