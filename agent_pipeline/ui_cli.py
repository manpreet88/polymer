"""Command line interface for the polymer AI agent pipeline."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from .config import CheckpointConfig, KnowledgeBaseConfig
from .data_ingestion import PolymerDataIngestionTool
from .embedding_service import MultimodalEmbeddingService
from .knowledge_base import PolymerKnowledgeBase
from .property_predictor import PropertyPredictorEnsemble

LOGGER = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def launch_cli(argv: Any = None) -> None:
    parser = argparse.ArgumentParser(description="Polymer AI agent toolkit")
    parser.add_argument("psmiles", help="Input PSMILES string for processing")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k", help="Number of neighbors to retrieve")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-save", action="store_true", help="Do not persist to the knowledge base")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    checkpoint_cfg = CheckpointConfig().resolve()
    kb_cfg = KnowledgeBaseConfig().resolve()

    ingestion_tool = PolymerDataIngestionTool()
    embedding_service = MultimodalEmbeddingService(checkpoint_cfg)
    knowledge_base = PolymerKnowledgeBase(kb_cfg)
    predictor = PropertyPredictorEnsemble(checkpoint_cfg, embedding_service=embedding_service)

    polymer = ingestion_tool.ingest_psmiles(args.psmiles, source="cli")
    embedding = embedding_service.embed_polymer(polymer)

    knowledge_base.add_entries([polymer], [embedding])
    if not args.no_save:
        knowledge_base.save()

    print("=== Processed Polymer ===")
    print(f"Canonical PSMILES: {polymer.psmiles}")

    neighbors = knowledge_base.search(embedding.vector, top_k=args.top_k)
    if neighbors:
        print("\n=== Nearest Knowledge Base Entries ===")
        for entry, score in neighbors:
            print(f"{entry.psmiles} | similarity={score:.3f} | source={entry.source}")
    else:
        print("\nKnowledge base is empty after ingestion.")

    predictions = predictor.predict_from_embedding(embedding)
    if predictions:
        print("\n=== Property Predictions ===")
        for pred in predictions:
            unit = f" {pred.unit}" if pred.unit else ""
            print(f"{pred.name}: {pred.mean:.3f}{unit} ± {pred.std:.3f}")
    else:
        print("\nNo property heads available.")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    launch_cli()
