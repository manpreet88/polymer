"""Encoder tool abstractions that wrap pretrained modality checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

from ..config import ModelRegistryPaths

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for documentation builds
    torch = None  # type: ignore


class EncoderTool(Protocol):
    """Interface implemented by all modality-specific encoder tools."""

    name: str

    def encode(self, batch: Iterable[Any]) -> List[List[float]]:
        ...


@dataclass
class TorchEncoderTool:
    """Utility wrapper around a PyTorch module and checkpoint."""

    name: str
    module: Any
    checkpoint_path: Optional[Path]

    def encode(self, batch: Iterable[Any]) -> List[List[float]]:
        if torch is None:
            raise ImportError("PyTorch is required to run encoder tools")
        if self.checkpoint_path and self.checkpoint_path.exists():
            state = torch.load(self.checkpoint_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            self.module.load_state_dict(state, strict=False)
        self.module.eval()
        embeddings: List[List[float]] = []
        with torch.no_grad():
            for item in batch:
                tensor = self._to_tensor(item)
                output = self.module(tensor)
                if hasattr(output, "detach"):
                    output = output.detach()
                embeddings.append(output.flatten().tolist())
        return embeddings

    def _to_tensor(self, item: Any) -> Any:
        if torch is None:
            raise ImportError("PyTorch is required to run encoder tools")
        if isinstance(item, torch.Tensor):
            return item.unsqueeze(0) if item.ndim == 1 else item
        if isinstance(item, (list, tuple)):
            return torch.tensor(item)
        raise TypeError(f"Unsupported batch item type: {type(item)!r}")


class EncoderRegistry:
    """Factory that instantiates tools for each available modality."""

    def __init__(self, paths: ModelRegistryPaths) -> None:
        self._paths = paths

    def create_gine_tool(self) -> TorchEncoderTool:
        from gine import GINE

        model = GINE()
        return TorchEncoderTool("gine", model, self._paths.gine)

    def create_schnet_tool(self) -> TorchEncoderTool:
        from schnet import SchNet

        model = SchNet()
        return TorchEncoderTool("schnet", model, self._paths.schnet)

    def create_fingerprint_tool(self) -> TorchEncoderTool:
        from transformer import FingerprintTransformer

        model = FingerprintTransformer()
        return TorchEncoderTool("fingerprint", model, self._paths.fingerprint)

    def create_deberta_tool(self) -> TorchEncoderTool:
        from debertav2 import PolymerDebertaModel

        model = PolymerDebertaModel()
        return TorchEncoderTool("deberta", model, self._paths.deberta)

    def dump_registry_state(self) -> Dict[str, str]:
        return {name: str(path) for name, path in self._paths.as_dict().items()}

    def to_json(self) -> str:
        return json.dumps(self.dump_registry_state(), indent=2)
