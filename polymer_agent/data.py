"""Wrappers around the repository's multimodal data extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from Data_Modalities import AdvancedPolymerMultimodalExtractor
except ImportError:  # pragma: no cover - optional dependency fallback
    AdvancedPolymerMultimodalExtractor = None  # type: ignore


@dataclass
class MultimodalDataBundle:
    """Container for the heterogeneous features generated per polymer."""

    psmiles: str
    graph: Optional[Dict[str, Any]]
    geometry: Optional[Dict[str, Any]]
    fingerprint: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        """Return a metadata-only dictionary useful for logging."""

        return {
            "psmiles": self.psmiles,
            **{k: v for k, v in self.metadata.items() if k != "raw"},
        }


class MultimodalDataExtractor:
    """High-level helper that sanitizes input and fetches multimodal features.

    The class lazily instantiates :class:`AdvancedPolymerMultimodalExtractor`
    to avoid importing heavy scientific dependencies until necessary.
    """

    def __init__(self, **extractor_kwargs: Any) -> None:
        if AdvancedPolymerMultimodalExtractor is None:
            raise ImportError(
                "AdvancedPolymerMultimodalExtractor is unavailable. "
                "Ensure Data_Modalities dependencies (e.g., RDKit) are installed."
            )
        self._extractor = AdvancedPolymerMultimodalExtractor(**extractor_kwargs)

    def from_psmiles(self, psmiles: str) -> MultimodalDataBundle:
        """Generate a multimodal bundle from a pSMILES string."""

        result = self._extractor([psmiles])[0]
        metadata = result.get("metadata", {})
        return MultimodalDataBundle(
            psmiles=psmiles,
            graph=result.get("graph"),
            geometry=result.get("geometry"),
            fingerprint=result.get("fingerprint"),
            metadata=metadata,
        )

    def batch_from_psmiles(self, smiles_list: Any) -> Dict[str, MultimodalDataBundle]:
        """Batch version of :meth:`from_psmiles` returning a dictionary."""

        outputs: Dict[str, MultimodalDataBundle] = {}
        for entry in self._extractor(smiles_list):
            psmiles = entry.get("psmiles") or entry.get("metadata", {}).get("psmiles")
            if not psmiles:
                raise ValueError("Extractor returned an entry without a pSMILES identifier")
            outputs[psmiles] = MultimodalDataBundle(
                psmiles=psmiles,
                graph=entry.get("graph"),
                geometry=entry.get("geometry"),
                fingerprint=entry.get("fingerprint"),
                metadata=entry.get("metadata", {}),
            )
        return outputs
