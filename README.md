# Polymer Multimodal Agent Framework

This repository now includes a GPT-4 orchestrated multi-agent framework that
automates literature review, data curation, property modeling, and generative
planning for the polymer foundational model stack.

## Features

- **GPT-4 Orchestrator** routes requests across specialist agents.
- **Specialist Agents** for literature research, data curation, property
  prediction, polymer generation, and technical visualisation.
- **Command Line Interface** to run end-to-end orchestration with contextual
  metadata about your dataset and goals.

## Installation

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Export your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## Usage

Run the orchestrator with a natural language request:

```bash
python -m examples.run_polymer_agent \
  "Design a closed-loop workflow to discover biodegradable polymer carriers for oncology drugs" \
  --constraints budget=low compute=8xA100 \
  --notes timeline=6months
```

The command prints a JSON object containing the execution plan, each specialist
output, and the orchestrated final response. Use `--output plan.json` to store
the results on disk.

### Example Requests

1. **Data onboarding**
   ```bash
   python -m examples.run_polymer_agent "Curate a new dataset of polymer-drug conjugates for contrastive pretraining"
   ```

2. **Property optimization**
   ```bash
   python -m examples.run_polymer_agent "Improve Tg prediction accuracy for amphiphilic block copolymers"
   ```

3. **Generative design loop**
   ```bash
   python -m examples.run_polymer_agent "Plan a generation-study-evaluate loop for tumor-targeted degradable polymers"
   ```

### Generate Technical Visuals for New Polymers

Create visual summaries (molecule grids, descriptor tables, and plots) from
generated SMILES using the built-in toolkit:

```bash
python -m examples.visualize_generated_polymers \
  --smiles-file examples/sample_generated_smiles.json \
  --output-dir outputs/visual_report
```

Alternatively, append `--visualize-smiles` arguments to the orchestrator CLI to
render visuals automatically after an agent run:

```bash
python -m examples.run_polymer_agent \
  "Design polymer carriers for targeted siRNA delivery" \
  --visualize-smiles examples/sample_generated_smiles.json \
  --visual-output-dir outputs/visual_from_orchestrator
```

## Extending

Add new specialists by subclassing `BaseLLMAgent` and injecting them into
`build_default_orchestrator`. The orchestrator automatically includes them in
its planning roster.

For deterministic tooling (like the provided
`PolymerVisualizationToolkit`), expose helper functions under `agents/` and have
specialists reference them so the orchestrator can tie LLM guidance to concrete
code.

## Repository Scripts

The original training scripts (`Data_Modalities.py`, `CL.py`, `Polymer_Generation.py`,
etc.) remain unchanged and can be called from the workflows proposed by the
agents.
