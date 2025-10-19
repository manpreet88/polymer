"""High-level pipeline assembly helpers inspired by oncology AI-agent workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import Draw

from .config import CheckpointConfig, KnowledgeBaseConfig
from .data_ingestion import PolymerDataIngestionTool
from .datamodels import GeneratedPolymer, ProcessedPolymer, PropertyPrediction
from .embedding_service import MultimodalEmbeddingService
from .generation import PolymerGeneratorTool
from .knowledge_base import PolymerKnowledgeBase
from .knowledge_sources import (
    build_ingestion_payloads,
    read_reference_catalog,
)
from .property_predictor import PropertyPredictorEnsemble

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineServices:
    """Bundle of core services that mirror the oncology agent stack."""

    ingestion: PolymerDataIngestionTool
    embedding: MultimodalEmbeddingService
    knowledge_base: PolymerKnowledgeBase
    predictor: PropertyPredictorEnsemble
    generator: PolymerGeneratorTool


def build_pipeline_services(
    *,
    checkpoint_cfg: Optional[CheckpointConfig] = None,
    knowledge_base_cfg: Optional[KnowledgeBaseConfig] = None,
) -> PipelineServices:
    """Construct the aligned ingestion/embedding/prediction/generation services."""

    checkpoint_cfg = (checkpoint_cfg or CheckpointConfig()).resolve()
    kb_cfg = (knowledge_base_cfg or KnowledgeBaseConfig()).resolve()

    ingestion = PolymerDataIngestionTool()
    embedding = MultimodalEmbeddingService(checkpoint_cfg)
    knowledge_base = PolymerKnowledgeBase(kb_cfg)
    predictor = PropertyPredictorEnsemble(checkpoint_cfg, embedding_service=embedding)
    generator = PolymerGeneratorTool(embedding, knowledge_base, predictor)
    return PipelineServices(
        ingestion=ingestion,
        embedding=embedding,
        knowledge_base=knowledge_base,
        predictor=predictor,
        generator=generator,
    )


def seed_knowledge_base_from_catalog(
    services: PipelineServices,
    catalog_path: Path,
    *,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> int:
    """Populate the knowledge base using curated polymer catalogs."""

    records = read_reference_catalog(catalog_path)
    if limit is not None:
        records = records[:limit]
    payloads = build_ingestion_payloads(records)

    if overwrite:
        services.knowledge_base.entries.clear()

    processed: List[ProcessedPolymer] = []
    embeddings = []
    for payload in payloads:
        psmiles = payload["psmiles"]
        if not overwrite and services.knowledge_base.contains_psmiles(psmiles):
            LOGGER.debug("Skipping existing PSMILES %s", psmiles)
            continue
        metadata = payload.get("metadata", {})
        polymer = services.ingestion.ingest_psmiles(
            psmiles,
            source=payload.get("source", "reference"),
            metadata=metadata,
        )
        processed.append(polymer)
        embeddings.append(services.embedding.embed_polymer(polymer))
    if processed:
        services.knowledge_base.add_entries(processed, embeddings)
        services.knowledge_base.save()
    LOGGER.info("Seeded knowledge base with %d catalog polymers", len(processed))
    return len(processed)


def summarise_predictions(predictions: Sequence[PropertyPrediction]) -> Dict[str, Dict[str, float]]:
    """Return mean/std summary keyed by property name."""

    summary: Dict[str, Dict[str, float]] = {}
    for prediction in predictions:
        summary[prediction.name] = {
            "mean": prediction.mean,
            "std": prediction.std,
            "unit": prediction.unit,
        }
    return summary


def run_non_expert_walkthrough(
    services: PipelineServices,
    *,
    psmiles: str,
    property_focus: Optional[Sequence[str]] = None,
    neighbours: int = 3,
) -> Dict[str, object]:
    """Simulate a non-expert experience similar to the oncology agent paper."""

    polymer = services.ingestion.ingest_psmiles(psmiles, source="non_expert")
    embedding = services.embedding.embed_polymer(polymer)
    predictions = services.predictor.predict_from_embedding(embedding)
    if property_focus:
        predictions = [pred for pred in predictions if pred.name in set(property_focus)]
    neighbours_found = services.knowledge_base.search(embedding.vector, top_k=neighbours)

    return {
        "psmiles": polymer.psmiles,
        "predictions": summarise_predictions(predictions),
        "neighbours": [
            {
                "psmiles": entry.psmiles,
                "source": entry.source,
                "similarity": score,
                "name": entry.metadata.get("name"),
            }
            for entry, score in neighbours_found
        ],
    }


def run_expert_design_round(
    services: PipelineServices,
    *,
    design_brief: str,
    seed_psmiles: str,
    candidate_count: int = 5,
    mutations: int = 2,
    image_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Mirror an expert-facing round with retrieval, generation, and visualisation."""

    polymer = services.ingestion.ingest_psmiles(seed_psmiles, source="expert")
    embedding = services.embedding.embed_polymer(polymer)

    candidates: List[GeneratedPolymer] = services.generator.generate(
        polymer, top_k=candidate_count, mutations=mutations
    )

    visualisations: Dict[str, str] = {}
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            path = image_dir / f"candidate_{candidate.psmiles.replace('/', '_')}.png"
            _render_psmiles(candidate.psmiles, path)
            visualisations[candidate.psmiles] = str(path)

    candidate_payloads = []
    for candidate in candidates:
        predictions = services.predictor.predict_polymer(
            services.ingestion.ingest_psmiles(candidate.psmiles, source="candidate")
        )
        candidate_payloads.append(
            {
                "psmiles": candidate.psmiles,
                "generation_mode": candidate.metadata.get("mode"),
                "properties": summarise_predictions(predictions),
                "visual": visualisations.get(candidate.psmiles),
            }
        )

    return {
        "brief": design_brief,
        "seed": polymer.psmiles,
        "seed_neighbours": [
            {
                "psmiles": entry.psmiles,
                "source": entry.source,
                "name": entry.metadata.get("name"),
                "similarity": score,
            }
            for entry, score in services.knowledge_base.search(embedding.vector, top_k=5)
        ],
        "candidates": candidate_payloads,
    }


def _render_psmiles(psmiles: str, path: Path) -> None:
    """Render a PSMILES string to a PNG file using RDKit."""

    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        LOGGER.warning("Unable to render PSMILES %s", psmiles)
        return
    Draw.MolToFile(mol, str(path))


__all__ = [
    "PipelineServices",
    "build_pipeline_services",
    "seed_knowledge_base_from_catalog",
    "run_non_expert_walkthrough",
    "run_expert_design_round",
]
