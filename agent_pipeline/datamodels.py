"""Common dataclasses shared across the polymer AI agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class PolymerModalities:
    """Container for the multimodal representations of a polymer."""

    graph: Dict
    geometry: Dict
    fingerprints: Dict
    extra: Dict = field(default_factory=dict)


@dataclass
class ProcessedPolymer:
    """Standardized polymer entry produced by the ingestion tool."""

    psmiles: str
    source: str
    modalities: PolymerModalities
    metadata: Dict


@dataclass
class EmbeddingResult:
    """Embedding output for a polymer."""

    vector: np.ndarray
    modalities_used: List[str]
    reconstruction_losses: Dict[str, float]


@dataclass
class PropertyPrediction:
    """Prediction for a polymer property."""

    name: str
    mean: float
    std: float
    unit: Optional[str] = None
    raw_outputs: Dict[str, float] = field(default_factory=dict)


@dataclass
class GeneratedPolymer:
    """Polymer produced by the generative tool."""

    psmiles: str
    metadata: Dict
    property_estimates: List[PropertyPrediction] = field(default_factory=list)


@dataclass
class ToolInvocationLog:
    """Audit log entry for a tool invocation."""

    tool_name: str
    arguments: Dict
    timestamp: datetime
    outputs_preview: str


def summarize_predictions(predictions: Iterable[PropertyPrediction]) -> Dict[str, float]:
    """Convert an iterable of predictions into a dictionary summary."""

    summary: Dict[str, float] = {}
    for pred in predictions:
        summary[pred.name] = pred.mean
    return summary


__all__ = [
    "PolymerModalities",
    "ProcessedPolymer",
    "EmbeddingResult",
    "PropertyPrediction",
    "GeneratedPolymer",
    "ToolInvocationLog",
    "summarize_predictions",
]
