"""Specialist agent definitions."""
from __future__ import annotations

from .base import AgentContext, BaseLLMAgent
from .config import AgentConfig, ModelConfig


DEFAULT_SPECIALIST_MODEL = ModelConfig(model="gpt-4o-mini", temperature=0.1)


class LiteratureResearchAgent(BaseLLMAgent):
    """Agent dedicated to literature triage and synthesis."""

    def __init__(self, *, client) -> None:
        super().__init__(
            config=AgentConfig(
                name="LiteratureResearchAgent",
                system_prompt=(
                    "You are a polymer science research assistant. Combine oncology agent "
                    "and polymer delivery insights. Prioritize recent peer-reviewed "
                    "findings, cite sources with DOIs when possible, and highlight "
                    "actionable hypotheses or validation studies."
                ),
                model_config=DEFAULT_SPECIALIST_MODEL,
            ),
            client=client,
        )

    def analyze(self, question: str, context: AgentContext) -> str:
        prompt = (
            "Provide a concise research brief that answers the question. Include: "
            "(1) key findings, (2) oncology relevance, (3) suggested experiments or "
            "datasets for validation."
        )
        task = f"Question: {question}\n\n{prompt}"
        return self.run(task=task, context=context)


class DataCurationAgent(BaseLLMAgent):
    """Agent that proposes data standardisation and QA plans."""

    def __init__(self, *, client) -> None:
        super().__init__(
            config=AgentConfig(
                name="DataCurationAgent",
                system_prompt=(
                    "You are responsible for polymer dataset QA, cleaning, and "
                    "augmentation. Design pipelines leveraging the existing tooling "
                    "in this repository, including Data_Modalities.py and downstream "
                    "pretraining scripts."
                ),
                model_config=DEFAULT_SPECIALIST_MODEL,
            ),
            client=client,
        )

    def design_pipeline(self, requirement: str, context: AgentContext) -> str:
        task = (
            "Outline a step-by-step data curation workflow. Reference repository "
            "scripts when relevant and provide shell commands that the user can run. "
            "Highlight validation checks for each modality."
        )
        return self.run(task=f"Requirement: {requirement}\n\n{task}", context=context)


class PropertyPredictionAgent(BaseLLMAgent):
    """Agent that recommends downstream property modeling strategies."""

    def __init__(self, *, client) -> None:
        super().__init__(
            config=AgentConfig(
                name="PropertyPredictionAgent",
                system_prompt=(
                    "You help design polymer property prediction experiments. "
                    "Leverage pretrained encoders and Property_Prediction.py. "
                    "Recommend metrics, validation splits, and uncertainty estimation."
                ),
                model_config=DEFAULT_SPECIALIST_MODEL,
            ),
            client=client,
        )

    def propose_experiments(self, goal: str, context: AgentContext) -> str:
        task = (
            "Describe the modeling approach, feature fusion strategy, and evaluation "
            "protocol for the stated goal. Provide specific commands and configuration "
            "edits when possible."
        )
        return self.run(task=f"Goal: {goal}\n\n{task}", context=context)


class PolymerGenerationAgent(BaseLLMAgent):
    """Agent that guides polymer generative modeling."""

    def __init__(self, *, client) -> None:
        super().__init__(
            config=AgentConfig(
                name="PolymerGenerationAgent",
                system_prompt=(
                    "You design polymer generative model experiments using the "
                    "multimodal contrastive encoders and Polymer_Generation.py. "
                    "Advise on conditioning signals, decoding strategies, and "
                    "post-generation filtering (e.g., RDKit validation)."
                ),
                model_config=DEFAULT_SPECIALIST_MODEL,
            ),
            client=client,
        )

    def plan_generation(self, specification: str, context: AgentContext) -> str:
        task = (
            "Propose how to generate polymers matching the specification. Include "
            "hyperparameters, conditioning features, and evaluation metrics."
        )
        return self.run(task=f"Specification: {specification}\n\n{task}", context=context)


class VisualizationAgent(BaseLLMAgent):
    """Agent that recommends technical visualisation workflows."""

    def __init__(self, *, client) -> None:
        super().__init__(
            config=AgentConfig(
                name="VisualizationAgent",
                system_prompt=(
                    "You design polymer visualisation studies. Reference the helper "
                    "module `agents.visualization.PolymerVisualizationToolkit` for "
                    "generating molecule grids, descriptor tables, and plots. Provide "
                    "step-by-step commands and explain how to interpret the visuals."
                ),
                model_config=DEFAULT_SPECIALIST_MODEL,
            ),
            client=client,
        )

    def plan_visualisations(self, requirement: str, context: AgentContext) -> str:
        task = (
            "Outline a visual analytics workflow covering molecule renderings, "
            "descriptor computation, and property plots. Mention file outputs and "
            "how to link them back to generated polymers."
        )
        return self.run(task=f"Requirement: {requirement}\n\n{task}", context=context)
