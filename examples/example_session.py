"""Example end-to-end usage of the polymer AI agent pipeline."""

from __future__ import annotations

import logging

from agent_pipeline import (
    CheckpointConfig,
    KnowledgeBaseConfig,
    PolymerDataIngestionTool,
    MultimodalEmbeddingService,
    PolymerKnowledgeBase,
    PropertyPredictorEnsemble,
    PolymerGeneratorTool,
    ToolRegistry,
    GPT4Orchestrator,
)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration and service construction
# ---------------------------------------------------------------------------
checkpoint_cfg = CheckpointConfig().resolve()
kb_cfg = KnowledgeBaseConfig().resolve()

ingestion = PolymerDataIngestionTool()
embedding_service = MultimodalEmbeddingService(checkpoint_cfg)
knowledge_base = PolymerKnowledgeBase(kb_cfg)
predictor = PropertyPredictorEnsemble(checkpoint_cfg, embedding_service=embedding_service)
generator = PolymerGeneratorTool(embedding_service, knowledge_base, predictor)

# ---------------------------------------------------------------------------
# Ingest polymers from a simple list (replace with CSV ingestion if needed)
# ---------------------------------------------------------------------------
polymer_smiles_list = ["[*]CCO", "[*]CCN", "[*]CCC(=O)O"]
processed = [ingestion.ingest_psmiles(ps) for ps in polymer_smiles_list]
embeddings = [embedding_service.embed_polymer(p) for p in processed]
knowledge_base.add_entries(processed, embeddings)
knowledge_base.save()

# ---------------------------------------------------------------------------
# Retrieve neighbours & run property predictions
# ---------------------------------------------------------------------------
seed_polymer = processed[0]
seed_embedding = embeddings[0]
print("Nearest neighbours: ")
for entry, score in knowledge_base.search(seed_embedding.vector, top_k=3):
    print(f"  {entry.psmiles} | similarity={score:.3f}")

print("\nPredicted properties:")
for prediction in predictor.predict_from_embedding(seed_embedding):
    unit = f" {prediction.unit}" if prediction.unit else ""
    print(f"  {prediction.name}: {prediction.mean:.3f}{unit} ± {prediction.std:.3f}")

# ---------------------------------------------------------------------------
# Generate new candidates via retrieval + mutation
# ---------------------------------------------------------------------------
print("\nGenerated candidates:")
for candidate in generator.generate(seed_polymer, top_k=2, mutations=2):
    print(f"  {candidate.metadata['mode']} -> {candidate.psmiles}")

# ---------------------------------------------------------------------------
# Wire services into GPT-4 orchestrator (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------
registry = ToolRegistry()
registry.register(
    "embed_polymer",
    "Embed a PSMILES string using the multimodal model",
    lambda args: embedding_service.embed_polymer(ingestion.ingest_psmiles(args["psmiles"])).vector.tolist(),
    parameters={
        "type": "object",
        "properties": {"psmiles": {"type": "string"}},
        "required": ["psmiles"],
    },
)
registry.register(
    "search_knowledge_base",
    "Retrieve polymers similar to a query embedding",
    lambda args: [
        {"psmiles": entry.psmiles, "similarity": score}
        for entry, score in knowledge_base.search(args["vector"], top_k=args.get("top_k", 5))
    ],
    parameters={
        "type": "object",
        "properties": {
            "vector": {"type": "array", "items": {"type": "number"}},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["vector"],
    },
)
registry.register(
    "predict_properties",
    "Predict properties for a PSMILES string",
    lambda args: [
        {
            "name": pred.name,
            "mean": pred.mean,
            "std": pred.std,
            "unit": pred.unit,
        }
        for pred in predictor.predict_polymer(ingestion.ingest_psmiles(args["psmiles"]))
    ],
    parameters={
        "type": "object",
        "properties": {"psmiles": {"type": "string"}},
        "required": ["psmiles"],
    },
)

try:
    orchestrator = GPT4Orchestrator(registry=registry)
    answer = orchestrator.chat(
        "Suggest polymers with higher thermal stability than [*]CCO and explain the reasoning."
    )
    print("\nGPT-4 orchestrator reply:\n", answer)
except Exception as exc:
    print("Skipping GPT-4 orchestrator demo:", exc)
