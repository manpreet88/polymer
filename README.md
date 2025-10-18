Polymer Foundation Models
==========================

This repository hosts multimodal encoders for polymer research, including graph,
3D, fingerprint, and PSMILES transformers, trained via contrastive learning.

Polymer AI Agent
----------------

The `Polymer_Agent.py` module introduces a Retrieval-Augmented Generation (RAG)
agent modeled after the "AI Agent for Oncology" Nature workflow. The agent
accepts polymer SMILES/PSMILES strings, aligns them with the multimodal
contrastive encoders from `CL.py`, and orchestrates evidence-grounded reasoning
through GPT-4. The system is usable by polymer experts and non-experts alike.

### Architecture Overview

1. **ContrastivePolymerEncoder** – loads the multimodal contrastive checkpoint to
   embed polymer inputs into a shared latent space.
2. **PolymerKnowledgeBase** – maintains a vector index of curated polymer
   literature, lab logs, and structured datasets.
3. **PolymerEvidenceSynthesizer** – reranks retrieved evidence with a
   cross-encoder and formats it for GPT-4 consumption.
4. **GPT4Orchestrator** – supervises prompt construction, GPT-4 inference, and
   iterative refinement stages.
5. **PolymerAgent** – exposes high-level APIs (`analyze`, `plan_experiments`) for
   expert-grade analysis and accessible action plans.

### End-to-End Setup

1. **Prepare multimodal checkpoints**
   - Train encoders with `gine.py`, `schnet.py`, `transformer.py`, and
     `debertav2.py`.
   - Align them using `CL.py` and save the fused checkpoint to
     `./checkpoints/polymer_cl/contrastive.pt`.

2. **Curate knowledge base artifacts**
   - Aggregate polymer literature, experiment logs, simulation results, and
     process guidelines.
   - Structure each record as JSON containing `psmiles`, `title`, `summary`,
     `experiment`, and `source` fields.

3. **Bootstrap the knowledge base**
   - Run `Polymer_Agent.bootstrap_knowledge_base` with the curated records to
     populate the vector store stored under `./artifacts/knowledge_base`.

4. **Configure GPT-4 access**
   - Export `OPENAI_API_KEY` or provide the key in `GPT4Config`.
   - Optionally adjust the GPT-4 model, temperature, and token budget.

5. **Run the agent**
   - Execute `python Polymer_Agent.py` to see expert and non-expert usage flows.
   - Integrate `PolymerAgent` into notebooks, web services, or lab automation
     pipelines via the Python API.

### Example Usage

```bash
export OPENAI_API_KEY=sk-...
python Polymer_Agent.py
```

The script prints:

- An expert-focused report synthesizing electrolyte design guidance.
- A non-expert onboarding brief with actionable lab instructions.

### Extending the Agent

- Plug in additional modalities (e.g., rheology measurements) by extending
  `PolymerKnowledgeBase.ingest` and the CL encoder projection heads.
- Integrate laboratory information management systems (LIMS) by appending new
  document sources during knowledge base ingestion.
- Combine with `Polymer_Generation.py` for closed-loop design and validation
  workflows.
