"""Utilities for loading pretrained encoders and tokenizers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .config import CheckpointConfig
from .models import (
    FingerprintEncoder,
    GineEncoder,
    MultimodalContrastiveModel,
    NodeSchNetWrapper,
    PSMILESDebertaEncoder,
    SimplePSMILESTokenizer,
)

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from safetensors.torch import load_file as load_safetensor
except Exception:  # pragma: no cover - optional dependency
    load_safetensor = None

try:  # pragma: no cover - optional dependency
    from transformers import DebertaV2Tokenizer
except Exception:  # pragma: no cover - fallback if transformers missing
    DebertaV2Tokenizer = None


def _resolve_checkpoint_path(directory: Path) -> Optional[Path]:
    """Return the first available checkpoint file within a directory."""

    candidates = ["pytorch_model.bin", "model.safetensors"]
    if directory.is_file():
        return directory
    for name in candidates:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _remap_state_keys(module: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Adjust checkpoint key names to match the lightweight inference modules."""

    def replace_prefix(mapping: Dict[str, str], tensor_map: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        updated: Dict[str, torch.Tensor] = {}
        for key, value in tensor_map.items():
            for old, new in mapping.items():
                if key.startswith(old):
                    key = new + key[len(old) :]
                    break
            updated[key] = value
        return updated

    if isinstance(module, GineEncoder):
        state = replace_prefix(
            {
                "layers.": "gnn_layers.",
                "_edge_to_node_proj": "edge_to_node",
                "edge_to_node_proj": "edge_to_node",
                "node_classifier.": "atom_head.",
            },
            state,
        )
    if isinstance(module, NodeSchNetWrapper):
        state = replace_prefix(
            {
                "base_schnet.": "schnet.base_schnet.",
            },
            state,
        )
    if isinstance(module, MultimodalContrastiveModel):
        state = replace_prefix(
            {
                "gine.layers.": "gine.gnn_layers.",
                "gine._edge_to_node_proj": "gine.edge_to_node",
                "gine.edge_to_node_proj": "gine.edge_to_node",
                "gine.node_classifier.": "gine.atom_head.",
                "schnet.base_schnet.": "schnet.schnet.base_schnet.",
                "schnet.schnet.base_schnet.": "schnet.schnet.base_schnet.",
                "fp.encoder.": "fp.",
            },
            state,
        )
    return state


def _load_state_dict(module: torch.nn.Module, checkpoint_path: Path) -> Tuple[int, int]:
    """Load a state dict from disk if it exists, returning missing/unexpected counts."""

    resolved_path = _resolve_checkpoint_path(checkpoint_path)
    if resolved_path is None:
        LOGGER.info("No checkpoint found at %s", checkpoint_path)
        return 0, 0
    if resolved_path.suffix == ".safetensors":
        if load_safetensor is None:
            raise RuntimeError(
                f"Encountered safetensors checkpoint at {resolved_path} but safetensors is not installed."
            )
        state: Dict[str, torch.Tensor] = load_safetensor(str(resolved_path))
    else:
        state = torch.load(resolved_path, map_location="cpu")

    if isinstance(module, FingerprintEncoder):
        encoder_prefix = "encoder."
        if any(key.startswith(encoder_prefix) for key in state.keys()):
            state = {key[len(encoder_prefix) :]: value for key, value in state.items() if key.startswith(encoder_prefix)}

    state = _remap_state_keys(module, state)

    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        LOGGER.warning("Missing keys for %s: %s", module.__class__.__name__, missing)
    if unexpected:
        LOGGER.warning("Unexpected keys for %s: %s", module.__class__.__name__, unexpected)
    return len(missing), len(unexpected)


def load_tokenizer(psmiles_dir: Path) -> Tuple[object, int]:
    """Load the tokenizer packaged with the PSMILES encoder if possible."""

    if DebertaV2Tokenizer is not None:
        try:
            tokenizer_path = None
            if (psmiles_dir / "tokenizer.json").exists():
                tokenizer_path = psmiles_dir
            elif (psmiles_dir / "tokenizer.model").exists():
                tokenizer_path = psmiles_dir / "tokenizer.model"
            if tokenizer_path is not None:
                tokenizer = DebertaV2Tokenizer.from_pretrained(str(psmiles_dir))
            else:
                tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
            tokenizer.add_special_tokens({"pad_token": "<pad>", "mask_token": "<mask>"})
            return tokenizer, len(tokenizer)
        except Exception as exc:  # pragma: no cover - fallback path
            LOGGER.warning("Falling back to simple tokenizer because HF tokenizer failed: %s", exc)
    tokenizer = SimplePSMILESTokenizer()
    return tokenizer, len(tokenizer)


def load_multimodal_model(
    checkpoints: CheckpointConfig,
    *,
    device: Optional[torch.device] = None,
    require_pretrained: bool = False,
) -> Tuple[MultimodalContrastiveModel, object]:
    """Instantiate encoders and load weights from the checkpoint bundle."""

    checkpoints = checkpoints.resolve()
    tokenizer, vocab_size = load_tokenizer(checkpoints.psmiles_dir)

    gine = None
    schnet = None
    fp = None
    psmiles = None

    try:
        gine = GineEncoder()
        _load_state_dict(gine, checkpoints.gine_dir)
    except Exception as exc:
        LOGGER.warning("Unable to initialize GINE encoder: %s", exc)
        if require_pretrained:
            raise

    try:
        schnet = NodeSchNetWrapper()
        _load_state_dict(schnet, checkpoints.schnet_dir)
    except Exception as exc:
        LOGGER.warning("Unable to initialize SchNet encoder: %s", exc)
        if require_pretrained:
            raise

    try:
        fp = FingerprintEncoder()
        _load_state_dict(fp, checkpoints.fingerprint_dir)
    except Exception as exc:
        LOGGER.warning("Unable to initialize fingerprint encoder: %s", exc)
        if require_pretrained:
            raise

    try:
        psmiles = PSMILESDebertaEncoder(checkpoints.psmiles_dir, tokenizer_vocab_size=vocab_size)
        _load_state_dict(psmiles, checkpoints.psmiles_dir)
    except Exception as exc:
        LOGGER.warning("Unable to initialize PSMILES encoder: %s", exc)
        if require_pretrained:
            raise

    model = MultimodalContrastiveModel(gine, schnet, fp, psmiles)
    _load_state_dict(model, checkpoints.multimodal_dir)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if gine is not None:
        gine.to(device)
    if schnet is not None:
        schnet.to(device)
    if fp is not None:
        fp.to(device)
    if psmiles is not None:
        psmiles.to(device)
    return model, tokenizer


__all__ = ["load_multimodal_model", "load_tokenizer"]
