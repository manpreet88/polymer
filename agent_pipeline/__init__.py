"""Polymer AI agent pipeline package."""

from .config import CheckpointConfig, KnowledgeBaseConfig, OpenAIConfig
from .data_ingestion import PolymerDataIngestionTool
from .embedding_service import MultimodalEmbeddingService
from .knowledge_base import PolymerKnowledgeBase
from .knowledge_sources import (
    CSVKnowledgeSource,
    JSONKnowledgeSource,
    PolymerKnowledgeSource,
    PolymerReferenceRecord,
    build_ingestion_payloads,
    load_reference_records,
    read_reference_catalog,
)
from .property_predictor import PropertyPredictorEnsemble
from .generation import PolymerGeneratorTool
from .orchestrator import GPT4Orchestrator, ToolRegistry
from .ui_cli import launch_cli
from .pipelines import (
    PipelineServices,
    build_pipeline_services,
    run_expert_design_round,
    run_non_expert_walkthrough,
    seed_knowledge_base_from_catalog,
)

__all__ = [
    "CheckpointConfig",
    "KnowledgeBaseConfig",
    "OpenAIConfig",
    "PolymerDataIngestionTool",
    "MultimodalEmbeddingService",
    "PolymerKnowledgeBase",
    "PolymerKnowledgeSource",
    "PolymerReferenceRecord",
    "CSVKnowledgeSource",
    "JSONKnowledgeSource",
    "load_reference_records",
    "build_ingestion_payloads",
    "read_reference_catalog",
    "PropertyPredictorEnsemble",
    "PolymerGeneratorTool",
    "GPT4Orchestrator",
    "ToolRegistry",
    "PipelineServices",
    "build_pipeline_services",
    "seed_knowledge_base_from_catalog",
    "run_non_expert_walkthrough",
    "run_expert_design_round",
    "launch_cli",
]
