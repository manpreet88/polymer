# Polymer AI Agent Pipeline

This repository packages the multimodal polymer foundation model into an interactive agent stack that mirrors the workflow of the "Development and validation of an autonomous artificial intelligence agent for clinical decision-making in oncology" paper, but adapted for polymer discovery. The pipeline wraps the existing contrastive learning encoders, downstream predictors, knowledge base, and generation utilities behind GPT-4 orchestration hooks so both experts and non-experts can explore polymers conversationally.

## Features

- **Data ingestion tool** – normalises user-supplied PSMILES and produces graph, geometry, and fingerprint modalities via `AdvancedPolymerMultimodalExtractor`.
- **Multimodal embedding service** – loads the contrastive learning encoders and projects polymers into a unified latent space.
- **Knowledge base / RAG** – stores embeddings plus metadata for similarity search, filtering, and retrieval-augmented reasoning.
- **Domain knowledge connectors** – ingest PoLyInfo, Polymer Genome, MatWeb, and handbook datasets via CSV/JSON catalogs for grounded retrieval.
- **Property prediction ensemble** – optional regression heads provide mean/std estimates for density, Tg, melting point, etc.
- **Generative assistant** – retrieves similar polymers and proposes lightweight RDKit mutations, evaluating them with the predictor.
- **GPT-4 orchestrator** – exposes all tools to a GPT-4 agent that can route tasks, call tools, and synthesise reports.
- **Command-line cockpit** – quick CLI for ingesting a polymer, updating the knowledge base, and running predictions.

## Repository layout

```
agent_pipeline/
    __init__.py                # Package exports
    config.py                  # Paths and API configuration
    datamodels.py              # Dataclasses used across tools
    data_ingestion.py          # Wrapper around AdvancedPolymerMultimodalExtractor
    embedding_service.py       # Multimodal encoder helper
    model_loader.py            # Checkpoint and tokenizer loading
    models.py                  # Inference-ready encoder definitions
    knowledge_base.py          # Simple vector store for RAG
    property_predictor.py      # Ensemble heads for property inference
    generation.py              # Retrieval + mutation generator
    orchestrator.py            # GPT-4 tool orchestrator
    ui_cli.py                  # CLI entry-point
examples/
    example_session.py         # End-to-end notebook-style usage
```

## Installation

1. Create and activate a Python 3.10+ environment.
2. Install dependencies (GPU support is optional but recommended):

   ```bash
   pip install -r requirements.txt
   ```

3. Place the pretrained checkpoints produced by the contrastive learning workflow in the expected directories (defaults shown below):

```
multimodal_output/best/pytorch_model.bin
gin_output/best/pytorch_model.bin
schnet_output/best/pytorch_model.bin
fingerprint_mlm_output/best/pytorch_model.bin
polybert_output/best/pytorch_model.bin
polybert_output/best/tokenizer.json (or tokenizer.model)
property_heads/<property_name>/*.pt (optional regression heads)
```

If you use different locations, instantiate `CheckpointConfig` with custom paths.

## GPT-4 access

Set your OpenAI credentials before running the orchestrator:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"          # or another GPT-4 family model
export OPENAI_ORG="org_..."            # optional
export OPENAI_TIMEOUT="120"            # optional request timeout in seconds
```

## Quick start (Python)

```python
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

checkpoint_cfg = CheckpointConfig().resolve()
kb_cfg = KnowledgeBaseConfig().resolve()

# Core services
ingestion = PolymerDataIngestionTool()
embedding_service = MultimodalEmbeddingService(checkpoint_cfg)
knowledge_base = PolymerKnowledgeBase(kb_cfg)
predictor = PropertyPredictorEnsemble(checkpoint_cfg, embedding_service=embedding_service)
generator = PolymerGeneratorTool(embedding_service, knowledge_base, predictor)

# Process a user polymer
polymer = ingestion.ingest_psmiles("[*]CCO")
embedding = embedding_service.embed_polymer(polymer)
knowledge_base.add_entries([polymer], [embedding])
knowledge_base.save()

# Retrieve neighbours and generate candidates
neighbours = knowledge_base.search(embedding.vector, top_k=3)
proposals = generator.generate(polymer, top_k=3, mutations=2)

# Register tools for GPT-4 orchestration
registry = ToolRegistry()
registry.register(
    "embed_polymer",
    "Embed a PSMILES using the multimodal model",
    lambda args: embedding_service.embed_polymer(ingestion.ingest_psmiles(args["psmiles"])).vector.tolist(),
    parameters={
        "type": "object",
        "properties": {"psmiles": {"type": "string"}},
        "required": ["psmiles"],
    },
)
registry.register(
    "search_knowledge_base",
    "Search the polymer knowledge base by vector similarity",
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

orchestrator = GPT4Orchestrator(registry=registry)
print(orchestrator.chat("Design a polymer similar to [*]CCO with higher thermal stability."))
```

## Command-line interface

Run the CLI to ingest and analyse a PSMILES:

```bash
python -m agent_pipeline.ui_cli "[*]CCO"
```

The command prints the canonicalised structure, nearest neighbours stored in the knowledge base, and any available property predictions. The knowledge base is persisted to `knowledge_base/` by default; pass `--no-save` to skip persistence.

## Example workflows

- `examples/example_session.py` demonstrates a lightweight notebook experience that:
  1. Builds the core services.
  2. Loads a batch of polymers from CSV and populates the knowledge base.
  3. Answers natural language questions via GPT-4 by wiring ingestion, embedding, retrieval, prediction, and generation tools together.
- `examples/high_level_scenarios.py` mirrors the oncology-agent paper with two personas:
  - **Non-expert triage** – a technician provides a rough PSMILES and immediately receives property predictions plus nearest neighbours retrieved from PoLyInfo/Polymer Genome/MatWeb references.
  - **Expert design studio** – a polymer chemist explores targeted candidates with automated retrieval, generator-driven mutations, quantitative scoring, and RDKit visualisations saved to `visualisations/`.
- `examples/paper_casebook.py` reproduces the detailed casebook flow described in the oncology agent publication, but adapted to polymers. It stages four contrasting personas (lab technician, process engineer, sustainability analyst, additive manufacturing specialist) and, for each:
  1. Normalises the provided PSMILES and embeds it with the multimodal encoder.
  2. Runs property predictions filtered to the persona’s objectives.
  3. Retrieves similar entries from the knowledge base.
  4. Optionally generates improved candidates and scores them.
  5. Calls the web RAG tool for external evidence.
  6. Synthesises a persona-aware Markdown report and saves any generated figures.

  Execute the casebook end-to-end with:

  ```bash
  python examples/paper_casebook.py
  ```

  The script resolves checkpoint and knowledge-base directories using `CheckpointConfig().resolve()` and `KnowledgeBaseConfig().resolve()`. Set the corresponding environment variables or edit `agent_pipeline/config.py` if your artefacts live elsewhere. On first run, it seeds the knowledge base from `examples/data/polymer_reference.csv` when available and prints a JSON summary for each scenario alongside the report markdown path. Ensure your OpenAI credentials are configured (see **GPT-4 access**) so the web RAG and report writer can call GPT-4.

Both scripts load a curated catalog located at `examples/data/polymer_reference.csv`. Replace this file with richer exports (PoLyInfo CSV, Polymer Genome JSONL, MatWeb scrapes, Polymer Handbook tables) to ground the knowledge base in authoritative sources. Use:

```python
from pathlib import Path
from agent_pipeline import build_pipeline_services, seed_knowledge_base_from_catalog

services = build_pipeline_services()
seed_knowledge_base_from_catalog(services, Path("polymer_catalog.csv"), overwrite=True)
```

The loader accepts CSV/TSV (columns prefixed with `property:` become structured numeric properties) and JSON/JSONL formats with arbitrary metadata.

## Provenance and safety

- Every tool invocation should be logged (set `logging` to `INFO` or `DEBUG`).
- The knowledge base saves metadata alongside embeddings to support audits.
- Property heads report ensemble mean and standard deviation for quick uncertainty assessment.
- Mutated polymers are evaluated through RDKit sanitisation; failures are logged and skipped.

## Extending the agent

- Swap the vector store for FAISS or Milvus by extending `PolymerKnowledgeBase`.
- Add new tools (e.g., retrosynthesis, regulatory screening) via `ToolRegistry.register` and expose them to GPT-4.
- Plug a richer front-end by reusing the services exposed in this package.

## Troubleshooting

- Missing checkpoints: ensure the `pytorch_model.bin` files exist at the configured paths. The loaders log informative warnings and keep operating with randomly initialised weights for experimentation.
- Torch Geometric / RDKit installation: refer to their official wheels for compatible CUDA versions.
- OpenAI errors: confirm API key, model name, and network connectivity; adjust `OPENAI_TIMEOUT` for long-running tool plans.

## License

Refer to the original repository's licensing terms for using the pretrained weights and data.
