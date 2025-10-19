"""Utilities for sourcing polymer reference data from domain-specific resources."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


@dataclass
class PolymerReferenceRecord:
    """Canonical representation of a polymer knowledge-base record.

    The fields map to the metadata highlighted throughout polymer informatics
    resources such as PoLyInfo, Polymer Genome, and MatWeb.  Only ``psmiles``
    and ``source`` are required; everything else is treated as metadata that is
    persisted alongside the embedding.
    """

    psmiles: str
    source: str
    name: Optional[str] = None
    reference: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    tags: Sequence[str] = field(default_factory=list)
    properties: Dict[str, float] = field(default_factory=dict)
    extra_metadata: Dict[str, object] = field(default_factory=dict)

    def to_ingestion_payload(self) -> Dict[str, object]:
        metadata = {
            "name": self.name,
            "reference": self.reference,
            "url": self.url,
            "description": self.description,
            "tags": list(self.tags),
            "properties": self.properties,
        }
        metadata.update(self.extra_metadata)
        return {
            "psmiles": self.psmiles,
            "source": self.source,
            "metadata": metadata,
        }


class PolymerKnowledgeSource:
    """Abstract interface for loading curated polymer reference records."""

    def load(self) -> List[PolymerReferenceRecord]:  # pragma: no cover - interface
        raise NotImplementedError


class CSVKnowledgeSource(PolymerKnowledgeSource):
    """Loads records from a CSV file with column-driven metadata."""

    def __init__(
        self,
        path: Path,
        *,
        source_column: str = "source",
        properties_prefix: str = "property:",
        tags_column: str = "tags",
    ) -> None:
        self.path = path
        self.source_column = source_column
        self.properties_prefix = properties_prefix
        self.tags_column = tags_column

    def load(self) -> List[PolymerReferenceRecord]:
        df = pd.read_csv(self.path)
        records: List[PolymerReferenceRecord] = []
        for _, row in df.iterrows():
            psmiles = row.get("psmiles")
            source = row.get(self.source_column, "unknown")
            if not isinstance(psmiles, str) or not psmiles:
                continue
            tags: Sequence[str] = []
            if self.tags_column in row and isinstance(row[self.tags_column], str):
                tags = [tag.strip() for tag in row[self.tags_column].split("|") if tag.strip()]
            properties: Dict[str, float] = {}
            for column in df.columns:
                if not column.startswith(self.properties_prefix):
                    continue
                value = row.get(column)
                if pd.isna(value):
                    continue
                properties[column[len(self.properties_prefix) :]] = float(value)
            extra_metadata: Dict[str, object] = {}
            for column in df.columns:
                if column in {"psmiles", self.source_column, self.tags_column}:
                    continue
                if column.startswith(self.properties_prefix):
                    continue
                value = row.get(column)
                if pd.isna(value):
                    continue
                extra_metadata[column] = value
            record = PolymerReferenceRecord(
                psmiles=psmiles,
                source=str(source),
                name=row.get("name"),
                reference=row.get("reference"),
                url=row.get("url"),
                description=row.get("description"),
                tags=tags,
                properties=properties,
                extra_metadata=extra_metadata,
            )
            records.append(record)
        return records


class JSONKnowledgeSource(PolymerKnowledgeSource):
    """Loads polymer reference records from a JSON or JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> List[PolymerReferenceRecord]:
        records: List[PolymerReferenceRecord] = []
        if self.path.suffix == ".jsonl":
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    records.append(self._record_from_mapping(payload))
        else:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("records", [])
            for entry in payload:
                records.append(self._record_from_mapping(entry))
        return records

    @staticmethod
    def _record_from_mapping(mapping: Dict) -> PolymerReferenceRecord:
        mapping = dict(mapping)
        properties = mapping.pop("properties", {}) or {}
        tags = mapping.pop("tags", []) or []
        extra_metadata = mapping.pop("metadata", {}) or {}
        return PolymerReferenceRecord(
            psmiles=mapping.pop("psmiles"),
            source=mapping.pop("source", "unknown"),
            name=mapping.pop("name", None),
            reference=mapping.pop("reference", None),
            url=mapping.pop("url", None),
            description=mapping.pop("description", None),
            tags=tags,
            properties=properties,
            extra_metadata={**mapping, **extra_metadata},
        )


def load_reference_records(sources: Iterable[PolymerKnowledgeSource]) -> List[PolymerReferenceRecord]:
    """Combine the records from the provided knowledge sources."""

    combined: List[PolymerReferenceRecord] = []
    for source in sources:
        combined.extend(source.load())
    return combined


def build_ingestion_payloads(records: Iterable[PolymerReferenceRecord]) -> List[Dict[str, object]]:
    """Convert reference records into ingestion-ready payload dictionaries."""

    return [record.to_ingestion_payload() for record in records]


def read_reference_catalog(path: Path) -> List[PolymerReferenceRecord]:
    """Load reference records from a CSV/JSON catalog based on file suffix."""

    path = path.expanduser().resolve()
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "," if path.suffix.lower() == ".csv" else "\t"
        if delimiter == ",":
            source = CSVKnowledgeSource(path)
            return source.load()
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)
        dataframe = pd.DataFrame(rows)
        temp_path = path.with_suffix(".csv")
        dataframe.to_csv(temp_path, index=False)
        source = CSVKnowledgeSource(temp_path)
        records = source.load()
        temp_path.unlink(missing_ok=True)
        return records
    if path.suffix.lower() in {".json", ".jsonl"}:
        return JSONKnowledgeSource(path).load()
    raise ValueError(f"Unsupported catalog format: {path.suffix}")


__all__ = [
    "PolymerReferenceRecord",
    "PolymerKnowledgeSource",
    "CSVKnowledgeSource",
    "JSONKnowledgeSource",
    "load_reference_records",
    "build_ingestion_payloads",
    "read_reference_catalog",
]
