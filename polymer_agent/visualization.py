"""Utilities for rendering polymer agent artifacts in text-friendly formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .retrieval import RetrievalResult


@dataclass
class TableColumn:
    name: str
    values: Sequence[object]


def format_table(columns: Iterable[TableColumn]) -> str:
    """Render a simple ASCII table suitable for console interfaces."""

    columns = list(columns)
    widths = [max(len(str(col.name)), *(len(str(v)) for v in col.values)) for col in columns]
    header = " | ".join(name.center(width) for name, width in zip((c.name for c in columns), widths))
    separator = "-+-".join("-" * width for width in widths)
    rows = []
    num_rows = max(len(col.values) for col in columns)
    for row_idx in range(num_rows):
        row_cells = []
        for col, width in zip(columns, widths):
            value = col.values[row_idx] if row_idx < len(col.values) else ""
            row_cells.append(str(value).ljust(width))
        rows.append(" | ".join(row_cells))
    return "\n".join([header, separator, *rows])


def render_retrieval_results(results: List[RetrievalResult]) -> str:
    """Return a human-readable representation of retrieval outputs."""

    if not results:
        return "No supporting context retrieved."
    payload = [result.to_message() for result in results]
    return "\n".join(f"- {item}" for item in payload)


def render_metadata(metadata: Dict[str, object]) -> str:
    """Pretty-print metadata dictionaries."""

    return json.dumps(metadata, indent=2, sort_keys=True)
