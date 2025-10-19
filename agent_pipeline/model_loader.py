"""Utilities for loading pretrained encoders and tokenizers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

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
    import sentencepiece as spm
except Exception:  # pragma: no cover - optional dependency
    spm = None

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


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Remove a common prefix from every key in the state dict."""

    if not any(key.startswith(prefix) for key in state):
        return state
    return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state.items()}


def _rename_prefix(state: Dict[str, torch.Tensor], old: str, new: str) -> Dict[str, torch.Tensor]:
    """Rename a single prefix in the state dict keys if present."""

    if not any(key.startswith(old) for key in state):
        return state
    return {
        (new + key[len(old) :] if key.startswith(old) else key): value for key, value in state.items()
    }


def _module_state_keys(module: torch.nn.Module) -> Sequence[str]:
    try:
        return tuple(module.state_dict().keys())
    except Exception:  # pragma: no cover - safety net for modules without state
        return ()


def _adapt_state_dict(module: torch.nn.Module, state: object) -> object:
    """Apply heuristic key renames so saved checkpoints line up with runtime modules."""

    if not isinstance(state, dict):
        return state

    adapted: Dict[str, torch.Tensor] = dict(state)

    if isinstance(module, FingerprintEncoder):
        adapted = _strip_prefix(adapted, "encoder.")
        if "mlm_head.weight" in adapted and "token_proj.weight" not in adapted:
            adapted["token_proj.weight"] = adapted.pop("mlm_head.weight")
        if "mlm_head.bias" in adapted and "token_proj.bias" not in adapted:
            adapted["token_proj.bias"] = adapted.pop("mlm_head.bias")

    if isinstance(module, GineEncoder):
        adapted = _rename_prefix(adapted, "gnn_layers.", "layers.")

    if isinstance(module, NodeSchNetWrapper):
        expected_keys = _module_state_keys(module)
        expects_base = any("base_schnet." in key for key in expected_keys)
        if expects_base:
            adapted = _rename_prefix(adapted, "schnet.", "schnet.base_schnet.")
        else:
            adapted = _rename_prefix(adapted, "schnet.base_schnet.", "schnet.")

    if isinstance(module, MultimodalContrastiveModel):
        adapted = _rename_prefix(adapted, "gine.gnn_layers.", "gine.layers.")
        if module.schnet is not None:
            schnet_expected = _module_state_keys(module.schnet)
            expects_base = any("base_schnet." in key for key in schnet_expected)
        else:
            expects_base = False
        if expects_base:
            adapted = _rename_prefix(adapted, "schnet.schnet.", "schnet.schnet.base_schnet.")
        else:
            adapted = _rename_prefix(adapted, "schnet.schnet.base_schnet.", "schnet.schnet.")
        adapted = _rename_prefix(adapted, "fp.encoder.", "fp.")
        if "fp.mlm_head.weight" in adapted and "fp.token_proj.weight" not in adapted:
            adapted["fp.token_proj.weight"] = adapted.pop("fp.mlm_head.weight")
        if "fp.mlm_head.bias" in adapted and "fp.token_proj.bias" not in adapted:
            adapted["fp.token_proj.bias"] = adapted.pop("fp.mlm_head.bias")

    return adapted


def _harmonize_prefixes(module: torch.nn.Module, state: object) -> object:
    """Attempt to strip or add prefixes so the state dict matches the module."""

    if not isinstance(state, dict):
        return state

    param_keys = list(module.state_dict().keys())
    if not param_keys:
        return state

    prefix_counts: Dict[str, int] = {}
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
                (key[len(prefix) :] if key.startswith(prefix) else key): value
                for key, value in state.items()
            }

    # Some checkpoints omit the "model." prefix for HuggingFace weights.
    if hasattr(module, "model") and isinstance(module.model, torch.nn.Module):
        needs_prefix = not any(key.startswith("model.") for key in state)
        if needs_prefix and any(
            key.startswith(prefix) for prefix in ("cls.", "deberta.", "encoder.", "embeddings.")
        ):
            state = {f"model.{key}": value for key, value in state.items()}

    return state


def _candidate_checkpoint_paths(base_path: Path) -> Iterable[Path]:
    """Yield plausible filenames for a checkpoint given a preferred path."""

    base_path = Path(base_path)

    yield base_path
    if base_path.suffix:
        yield base_path.with_suffix(".safetensors")

    parent_dirs = {base_path.parent}
    if base_path.parent.name == "best":
        parent_dirs.add(base_path.parent.parent)

    if base_path.is_dir():
        parent_dirs.add(base_path)

    candidate_names = (
        base_path.name,
        "pytorch_model.bin",
        "pytorch_model.pt",
        "pytorch_model.pth",
        "model.safetensors",
        "model.pt",
        "model.bin",
        "model.pth",
        "weights.pt",
        "checkpoint.pt",
        "checkpoint.pth",
        "best_model.pth",
        "best_model.pt",
        "best_model.bin",
        "best.ckpt",
    )

    for directory in parent_dirs:
        for name in candidate_names:
            if not name:
                continue
            yield directory / name


def load_checkpoint_state(module: torch.nn.Module, checkpoint_path: Path) -> Tuple[Sequence[str], Sequence[str]]:
    """Load weights into ``module`` from ``checkpoint_path`` if available."""

    if module is None:
        raise ValueError("module must not be None")

    checkpoint_path = Path(checkpoint_path)
    state: Optional[object] = None
    load_error: Optional[Exception] = None
    used_path: Optional[Path] = None

    for candidate in dict.fromkeys(_candidate_checkpoint_paths(checkpoint_path)):
        if not candidate.exists():
            continue
        if candidate.is_dir():
            continue
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
            LOGGER.debug("Failed to load checkpoint from %s: %s", candidate, exc)

    if state is None:
        if load_error is not None:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}") from load_error
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    LOGGER.info("Loaded checkpoint weights from %s", used_path)

    state = _unwrap_state_dict(state)
    state = _adapt_state_dict(module, state)
    state = _harmonize_prefixes(module, state)

    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        LOGGER.warning("Missing keys for %s: %s", module.__class__.__name__, missing)
    if unexpected:
        LOGGER.warning("Unexpected keys for %s: %s", module.__class__.__name__, unexpected)
    if not missing and not unexpected:
        LOGGER.info("All weights loaded for %s", module.__class__.__name__)

    return missing, unexpected


def load_tokenizer(psmiles_dir: Path) -> Tuple[object, int]:
    """Load the tokenizer packaged with the PSMILES encoder if possible."""

    psmiles_dir = Path(psmiles_dir)
    sp_path = psmiles_dir / "tokenizer.model"

    if sp_path.exists() and spm is not None:
        try:
            processor = spm.SentencePieceProcessor()
            processor.Load(str(sp_path))

            class SentencePieceTokenizer:
                def __init__(self, proc: spm.SentencePieceProcessor) -> None:
                    self._processor = proc
                    self.mask_token = "<mask>"
                    self.pad_token = "<pad>"
                    self.cls_token = "<cls>"
                    self.sep_token = "<sep>"
                    self.unk_token = "<unk>"
                    self.max_len = 128

                    self.mask_token_id = self._resolve_token_id(self.mask_token)
                    self.pad_token_id = self._resolve_token_id(self.pad_token)
                    self.cls_token_id = self._resolve_token_id(self.cls_token)
                    self.sep_token_id = self._resolve_token_id(self.sep_token)
                    self.unk_token_id = self._resolve_token_id(self.unk_token)

                def _resolve_token_id(self, token: str) -> int:
                    idx = self._processor.PieceToId(token)
                    if idx < 0:
                        idx = self._processor.unk_id()
                    return int(idx)

                def __len__(self) -> int:
                    return self._processor.GetPieceSize()

                def __call__(
                    self,
                    text: str,
                    *,
                    truncation: bool = True,
                    padding: str = "max_length",
                    max_length: Optional[int] = None,
                ) -> Dict[str, Sequence[int]]:
                    max_len = max_length or getattr(self, "max_len", 128)
                    ids = list(self._processor.EncodeAsIds(text))
                    if truncation:
                        ids = ids[:max_len]
                    attention = [1] * len(ids)
                    if padding == "max_length" and len(ids) < max_len:
                        pad_id = self.pad_token_id
                        pad_len = max_len - len(ids)
                        ids = ids + [pad_id] * pad_len
                        attention = attention + [0] * pad_len
                    return {"input_ids": ids, "attention_mask": attention}

            tokenizer = SentencePieceTokenizer(processor)
            tokenizer.model_max_length = getattr(tokenizer, "max_len", 128)
            LOGGER.info("Loaded SentencePiece tokenizer from %s", sp_path)
            print(f"Loaded tokenizer from {sp_path}")
            return tokenizer, len(tokenizer)
        except Exception as exc:  # pragma: no cover - safety fallback
            LOGGER.warning("Failed to load SentencePiece tokenizer: %s", exc)
    elif sp_path.exists():
        LOGGER.info("SentencePiece package not available; attempting Hugging Face tokenizer load")

    if DebertaV2Tokenizer is not None:
        try:
            tokenizer_path: Optional[Path] = None
            if (psmiles_dir / "tokenizer.json").exists():
                tokenizer_path = psmiles_dir
            elif sp_path.exists():
                tokenizer_path = sp_path
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
            LOGGER.info("Loaded fallback Hugging Face tokenizer")
            print("Loaded fallback Hugging Face tokenizer")
            return tokenizer, len(tokenizer)
        except Exception as exc:  # pragma: no cover - fallback path
            LOGGER.warning("Falling back to simple tokenizer because HF tokenizer failed: %s", exc)

    tokenizer = SimplePSMILESTokenizer()
    LOGGER.info("Using simple PSMILES tokenizer")
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

    gine: Optional[GineEncoder] = None
    schnet: Optional[NodeSchNetWrapper] = None
    fp: Optional[FingerprintEncoder] = None
    psmiles: Optional[PSMILESDebertaEncoder] = None

    try:
        gine = GineEncoder()
        load_checkpoint_state(gine, checkpoints.gine_dir / "pytorch_model.bin")
    except Exception as exc:
        LOGGER.warning("Unable to initialize GINE encoder: %s", exc)
        if require_pretrained:
            raise

    try:
        schnet = NodeSchNetWrapper()
        load_checkpoint_state(schnet, checkpoints.schnet_dir / "pytorch_model.bin")
    except Exception as exc:
        LOGGER.warning("Unable to initialize SchNet encoder: %s", exc)
        if require_pretrained:
            raise

    try:
        fp = FingerprintEncoder()
        load_checkpoint_state(fp, checkpoints.fingerprint_dir / "pytorch_model.bin")
    except Exception as exc:
        LOGGER.warning("Unable to initialize fingerprint encoder: %s", exc)
        if require_pretrained:
            raise

    try:
        psmiles = PSMILESDebertaEncoder(checkpoints.psmiles_dir, tokenizer_vocab_size=vocab_size)
        load_checkpoint_state(psmiles, checkpoints.psmiles_dir / "pytorch_model.bin")
    except Exception as exc:
        LOGGER.warning("Unable to initialize PSMILES encoder: %s", exc)
        if require_pretrained:
            raise

    model = MultimodalContrastiveModel(gine, schnet, fp, psmiles)
    mm_path = checkpoints.multimodal_dir / "pytorch_model.bin"
    load_checkpoint_state(model, mm_path)

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


__all__ = ["load_checkpoint_state", "load_multimodal_model", "load_tokenizer"]

