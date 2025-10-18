"""Utilities for loading pretrained encoders and tokenizers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

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
    from transformers import DebertaV2Tokenizer
except Exception:  # pragma: no cover - fallback if transformers missing
    DebertaV2Tokenizer = None


def _unwrap_state_dict(state: object) -> object:
    """Peel common wrappers (Lightning, DistributedDataParallel, etc.)."""

    if not isinstance(state, dict):
        return state
    for key in ("state_dict", "model_state_dict", "module", "model"):
        inner = state.get(key)
        if isinstance(inner, dict):
            state = inner
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        if any(key.startswith("module.") for key in state):
            state = {
                (key[len("module.") :] if key.startswith("module.") else key): value
                for key, value in state.items()
            }
    return state


def _harmonize_prefixes(module: torch.nn.Module, state: object) -> object:
    """Attempt to strip or add prefixes so the state dict matches the module."""

    if not isinstance(state, dict):
        return state

    param_keys = list(module.state_dict().keys())
    if not param_keys:
        return state

    prefix_counts = {}
    for key in state.keys():
        for param in param_keys:
            if key.endswith(param):
                prefix = key[: -len(param)]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
                break
    if prefix_counts:
        prefix, count = max(prefix_counts.items(), key=lambda item: item[1])
        if prefix and count >= max(1, len(param_keys) // 2):
            state = {
                (key[len(prefix) :] if key.startswith(prefix) else key): value for key, value in state.items()
            }

    # Some checkpoints omit the "model." prefix for HuggingFace weights.
    if hasattr(module, "model") and isinstance(module.model, torch.nn.Module):
        needs_prefix = not any(key.startswith("model.") for key in state)
        if needs_prefix and any(key.startswith(prefix) for prefix in ("cls.", "deberta.", "encoder.", "embeddings.")):
            state = {f"model.{key}": value for key, value in state.items()}

    return state


def _load_state_dict(module: torch.nn.Module, checkpoint_path: Path) -> Tuple[int, int]:
    """Load a state dict from disk if it exists, returning missing/unexpected counts."""

    if not checkpoint_path.exists():
        LOGGER.info("No checkpoint found at %s", checkpoint_path)
        return 0, 0
    state = torch.load(checkpoint_path, map_location="cpu")
    state = _unwrap_state_dict(state)
    state = _harmonize_prefixes(module, state)
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
        _load_state_dict(gine, checkpoints.gine_dir / "pytorch_model.bin")
    except Exception as exc:
        LOGGER.warning("Unable to initialize GINE encoder: %s", exc)
        if require_pretrained:
            raise

    try:
        schnet = NodeSchNetWrapper()
        _load_state_dict(schnet, checkpoints.schnet_dir / "pytorch_model.bin")
    except Exception as exc:
        LOGGER.warning("Unable to initialize SchNet encoder: %s", exc)
        if require_pretrained:
            raise

    try:
        fp = FingerprintEncoder()
        _load_state_dict(fp, checkpoints.fingerprint_dir / "pytorch_model.bin")
    except Exception as exc:
        LOGGER.warning("Unable to initialize fingerprint encoder: %s", exc)
        if require_pretrained:
            raise

    try:
        psmiles = PSMILESDebertaEncoder(checkpoints.psmiles_dir, tokenizer_vocab_size=vocab_size)
        _load_state_dict(psmiles, checkpoints.psmiles_dir / "pytorch_model.bin")
    except Exception as exc:
        LOGGER.warning("Unable to initialize PSMILES encoder: %s", exc)
        if require_pretrained:
            raise

    model = MultimodalContrastiveModel(gine, schnet, fp, psmiles)
    mm_path = checkpoints.multimodal_dir / "pytorch_model.bin"
    _load_state_dict(model, mm_path)

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
