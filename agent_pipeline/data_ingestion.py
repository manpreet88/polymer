"""Data ingestion utilities wrapping :mod:`Data_Modalities` for agent tooling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from Data_Modalities import AdvancedPolymerMultimodalExtractor

from .datamodels import PolymerModalities, ProcessedPolymer

LOGGER = logging.getLogger(__name__)


@dataclass
class IngestionSummary:
    """Summary returned after bulk ingestion."""

    processed: List[ProcessedPolymer]
    failures: List[Dict]


class PolymerDataIngestionTool:
    """Tool that converts raw PSMILES strings into aligned modalities."""

    def __init__(self, extractor: Optional[AdvancedPolymerMultimodalExtractor] = None) -> None:
        if extractor is None:
            # The extractor only needs a CSV path for batch utilities; single-sample usage works with a stub path.
            extractor = AdvancedPolymerMultimodalExtractor(csv_file="user_session.csv")
        self.extractor = extractor

    def ingest_psmiles(
        self,
        psmiles: str,
        *,
        source: str = "user",
        metadata: Optional[Dict] = None,
    ) -> ProcessedPolymer:
        """Process a single polymer PSMILES string."""

        metadata = metadata or {}
        canonical = self.extractor.validate_and_standardize_smiles(psmiles)
        if canonical is None:
            raise ValueError(f"Invalid PSMILES provided: {psmiles}")

        graph = self.extractor.generate_molecular_graph(canonical)
        geometry = self.extractor.optimize_3d_geometry(canonical)
        fingerprints = self.extractor.calculate_morgan_fingerprints(canonical)

        modalities = PolymerModalities(
            graph=graph,
            geometry=geometry,
            fingerprints=fingerprints,
            extra={"canonical_smiles": canonical},
        )
        processed = ProcessedPolymer(
            psmiles=canonical,
            source=source,
            modalities=modalities,
            metadata=metadata,
        )
        LOGGER.debug("Processed polymer %s", canonical)
        return processed

    def ingest_many(self, records: Iterable[Dict]) -> IngestionSummary:
        """Process an iterable of mapping objects with ``psmiles`` and ``source`` keys."""

        processed: List[ProcessedPolymer] = []
        failures: List[Dict] = []
        for idx, record in enumerate(records):
            smiles = record.get("psmiles")
            source = record.get("source", f"entry_{idx}")
            metadata = {k: v for k, v in record.items() if k not in {"psmiles", "source"}}
            try:
                processed.append(self.ingest_psmiles(smiles, source=source, metadata=metadata))
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to process record %s: %s", smiles, exc)
                failure = {"index": idx, "psmiles": smiles, "error": str(exc)}
                failures.append(failure)
        return IngestionSummary(processed=processed, failures=failures)


__all__ = ["PolymerDataIngestionTool", "IngestionSummary"]
