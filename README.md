# Polymer Multimodal Foundation Model Toolkit

This repository hosts scripts for extracting multimodal polymer descriptors, pretraining
contrastive encoders, and performing downstream tasks such as property prediction and
polymer generation. It now includes an advanced Retrieval-Augmented Generation (RAG)
assistant that lets domain experts and newcomers interact with the pretrained assets via
GPT-4 orchestration.

## New: `agent/` – Polymer RAG Agent

The `agent` package provides an ingest-retrieve-respond loop that reuses the saved
checkpoints from multimodal pretraining:

- **Multimodal ingestion** – Users provide a PSMILES string. The agent replays the
  validation and feature engineering pipeline from `Data_Modalities.py`, generating
  graphs, conformers, and fingerprint descriptors automatically.
- **Encoder-backed embeddings** – When available, the pretrained GINE, SchNet,
  fingerprint, and PSMILES encoders are loaded from `gin_output/best/`,
  `schnet_output/best/`, `fingerprint_mlm_output/best/`, and `polybert_output/best/`.
  Projection heads from `multimodal_output/best/` are used to project each modality into
  a shared latent space.
- **Hybrid knowledge base** – Encoded polymers are stored in an in-memory cosine-similarity
  index that supports hybrid retrieval across modalities.
- **GPT-4 orchestration** – Retrieved evidence is passed to GPT-4 (via the `openai`
  client) with a domain-specific system prompt that encourages grounded, citation-rich
  responses.
- **Visualization utilities** – Helpers render 2-D depictions (PNG) and optional 3-D
  Plotly scenes for quick inspection of candidate polymers.

### Package overview

```
agent/
├── agent.py              # PolymerRAGAgent orchestrating ingestion, retrieval, and GPT-4
├── config.py             # Dataclasses controlling checkpoints, retrieval, and LLM options
├── knowledge_base.py     # Simple cosine similarity index for multimodal embeddings
├── models.py             # Lightweight encoder definitions mirroring the training code
├── processing.py         # PSMILES sanitisation and tensor preparation helpers
├── visualization.py      # 2-D/3-D visualisation routines (RDKit + Plotly)
└── __init__.py           # Convenience exports
```

### Running the agent

1. Install dependencies (see below) and ensure the pretrained checkpoints are stored in
   the default directories or update `CheckpointConfig` accordingly.
2. Set the `OPENAI_API_KEY` environment variable.
3. Execute one of the examples, e.g. `python examples/basic_usage.py`.

The script ingests two polymers, retrieves similar entries for a query, and asks GPT-4 to
compare their flexibility while grounding the answer in retrieved descriptors.

#### Checkpoint locations

`CheckpointConfig` now scans several common locations for the exported weights. You can
leave the directory structure produced by the training scripts (`gin_output/`,
`schnet_output/`, etc.) at the repository root, or place the folders inside the `agent/`
package (e.g. `agent/gin_output/`). As long as a subdirectory contains a
`pytorch_model.bin` file it will be picked up automatically, so no manual configuration is
required when reusing existing checkpoints on a different machine.

If only a subset of checkpoints is available, the agent will keep operating with the
remaining encoders. For example, if the PSMILES model is missing (or its optional
`protobuf` dependency is not installed), the graph, fingerprint, and geometry encoders can
still serve retrieval requests.

### Visualisation example

Use `python examples/visualize.py` to produce a PNG rendering (always available) and an
interactive HTML view (requires Plotly). Outputs are saved to `./visualizations/`.

## Installation

```
pip install -r requirements.txt
```

GPU acceleration is optional but recommended when working with SchNet. When the agent
cannot load a particular encoder (e.g. missing `torch_geometric`), it will gracefully skip
that modality and continue operating with the remaining ones.

## Repository structure

- `Data_Modalities.py` – Multimodal feature extraction pipeline for polymer CSV datasets.
- `CL.py` – Multimodal contrastive pretraining script (original training entry point).
- `Polymer_Generation.py` – Generative fine-tuning leveraging pretrained encoders.
- `Property_Prediction.py` – Downstream property prediction using the shared encoders.
- `gine.py`, `schnet.py`, `transformer.py`, `debertav2.py` – Standalone encoder modules.
- `agent/` – Polymer RAG agent (described above).
- `examples/` – Usage demonstrations for the agent.

## Requirements

The minimal requirements for the agent are captured in `requirements.txt`. Depending on
which encoders you plan to load, you may also need `torch>=2.0` and `torch-geometric`
packages compatible with your CUDA setup.

## Contributing

Feel free to open issues or pull requests with enhancements to the agent (e.g. FAISS
backends, database persistence, additional downstream tools). When adding new modules,
consider documenting them here and providing a runnable example under `examples/`.
