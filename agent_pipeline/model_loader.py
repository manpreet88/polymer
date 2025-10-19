"""Utilities for loading pretrained encoders and tokenizers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

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
    from transformers import DebertaV2Tokenizer, PreTrainedTokenizerFast
except Exception:  # pragma: no cover - fallback if transformers missing
    DebertaV2Tokenizer = None
    PreTrainedTokenizerFast = None

try:  # pragma: no cover - optional dependency
    from safetensors.torch import load_file as load_safetensor
except Exception:  # pragma: no cover - safetensors is optional
    load_safetensor = None


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


def _strip_prefix(state: dict, prefix: str) -> dict:
    """Remove a common prefix from every key in the state dict."""

    if not any(key.startswith(prefix) for key in state):
        return state
    return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state.items()}


def _rename_prefix(state: dict, old: str, new: str) -> dict:
    """Rename a single prefix in the state dict keys if present."""

    if not any(key.startswith(old) for key in state):
        return state
    return {
        (new + key[len(old) :] if key.startswith(old) else key): value for key, value in state.items()
    }


def _adapt_state_dict(module: torch.nn.Module, state: object) -> object:
    """Apply heuristic key renames so saved checkpoints line up with runtime modules."""

    if not isinstance(state, dict):
        return state

    adapted = dict(state)

    if isinstance(module, FingerprintEncoder):
        adapted = _strip_prefix(adapted, "encoder.")
        if "mlm_head.weight" in adapted and "token_proj.weight" not in adapted:
            adapted["token_proj.weight"] = adapted.pop("mlm_head.weight")
        if "mlm_head.bias" in adapted and "token_proj.bias" not in adapted:
            adapted["token_proj.bias"] = adapted.pop("mlm_head.bias")

    if isinstance(module, GineEncoder):
        adapted = _rename_prefix(adapted, "gnn_layers.", "layers.")

    if isinstance(module, NodeSchNetWrapper):
        adapted = _rename_prefix(adapted, "schnet.base_schnet.", "schnet.")

    if isinstance(module, MultimodalContrastiveModel):
        adapted = _rename_prefix(adapted, "gine.gnn_layers.", "gine.layers.")
        adapted = _rename_prefix(adapted, "schnet.schnet.base_schnet.", "schnet.schnet.")
        adapted = _rename_prefix(adapted, "fp.encoder.", "fp.")
        if "fp.mlm_head.weight" in adapted and "fp.token_proj.weight" not in adapted:
            adapted["fp.token_proj.weight"] = adapted.pop("fp.mlm_head.weight")
        if "fp.mlm_head.bias" in adapted and "fp.token_proj.bias" not in adapted:
            adapted["fp.token_proj.bias"] = adapted.pop("fp.mlm_head.bias")

        needs_prefix = any(
            key.startswith(tuple(["cls.", "deberta.", "encoder.", "embeddings."])) for key in adapted
        ) and not any(key.startswith("model.") for key in adapted)
        if needs_prefix:
            adapted = {f"model.{key}": value for key, value in adapted.items()}

    return adapted


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


def _candidate_checkpoint_paths(base_path: Path) -> Iterable[Path]:
    """Yield plausible filenames for a checkpoint given a preferred path."""

    yield base_path
    if base_path.suffix:
        yield base_path.with_suffix(".safetensors")
    parent = base_path.parent
    for name in ("model.safetensors", "model.pt", "model.bin", "pytorch_model.bin"):
        yield parent / name


def _load_state_dict(module: torch.nn.Module, checkpoint_path: Path) -> Tuple[int, int]:
    """Load a state dict from disk if it exists, returning missing/unexpected counts."""

    load_error: Optional[Exception] = None
    state: Optional[object] = None
    used_path: Optional[Path] = None
    for candidate in dict.fromkeys(_candidate_checkpoint_paths(checkpoint_path)):
        if candidate.exists():
            try:
                if candidate.suffix == ".safetensors":
                    if load_safetensor is None:
                        raise RuntimeError("safetensors support is unavailable")
                    state = load_safetensor(str(candidate), device="cpu")
                else:
                    state = torch.load(candidate, map_location="cpu")
                used_path = candidate
                break
            except Exception as exc:  # pragma: no cover - IO issues are rare
                load_error = exc
    if state is None:
        if load_error is not None:
            LOGGER.warning("Failed to load checkpoint from %s: %s", checkpoint_path, load_error)
        else:
            LOGGER.info("No checkpoint found at %s", checkpoint_path)
        return 0, 0

    LOGGER.debug("Loaded checkpoint for %s from %s", module.__class__.__name__, used_path)
    state = _unwrap_state_dict(state)
    state = _adapt_state_dict(module, state)
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
                if (
                    isinstance(tokenizer_path, Path)
                    and tokenizer_path.is_file()
                    and PreTrainedTokenizerFast is not None
                ):
                    tokenizer = PreTrainedTokenizerFast(
                        tokenizer_file=str(tokenizer_path),
                        unk_token="<unk>",
                        pad_token="<pad>",
                        mask_token="<mask>",
                        cls_token="<cls>",
                        sep_token="<sep>",
                    )
                else:
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
