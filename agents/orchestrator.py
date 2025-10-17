"""GPT-4 orchestrator that coordinates specialist agents."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .base import AgentContext, BaseLLMAgent
from .clients import LLMMessage, OpenAIChatClient
from .config import AgentConfig, ModelConfig
from .specialists import (
    DataCurationAgent,
    LiteratureResearchAgent,
    PolymerGenerationAgent,
    PropertyPredictionAgent,
    VisualizationAgent,
)


ORCHESTRATOR_PROMPT = """You are the lead scientist for a polymer foundation model lab.
You delegate work to specialist agents. Follow this protocol:
1. Review the user request.
2. Decide which specialists should contribute (zero or more).
3. Draft a plan with numbered steps. Each step must specify the agent name and
a task string.
4. Return the plan as JSON with keys `rationale` and `steps` (a list with
`order`, `agent`, and `task`).
Only include agents that exist in the provided roster. Avoid free-form text.
"""

SUMMARY_PROMPT = """You have run the specialist plan. Synthesize their outputs into a
single actionable response for the user. Explicitly reference which agent
contributed each insight. Close with concrete next actions.
"""


@dataclass
class PlanStep:
    order: int
    agent: str
    task: str


class GPT4Orchestrator(BaseLLMAgent):
    """Agent that uses GPT-4 to route work across specialists."""

    def __init__(
        self,
        *,
        client: OpenAIChatClient,
        specialists: Sequence[BaseLLMAgent],
        model_config: ModelConfig = ModelConfig(model="gpt-4o", temperature=0.15),
    ) -> None:
        roster = {agent.name: agent for agent in specialists}
        if not roster:
            raise ValueError("At least one specialist agent is required.")
        self.specialists = roster

        plan_prompt = self._build_roster_prompt(roster)
        super().__init__(
            config=AgentConfig(
                name="GPT4Orchestrator",
                system_prompt=plan_prompt,
                model_config=model_config,
            ),
            client=client,
        )

    @staticmethod
    def _build_roster_prompt(roster: Dict[str, BaseLLMAgent]) -> str:
        roster_lines = [ORCHESTRATOR_PROMPT, "\nSpecialist roster:"]
        for name in roster:
            roster_lines.append(f"- {name}")
        return "\n".join(roster_lines)

    def create_plan(self, request: str, context: AgentContext) -> List[PlanStep]:
        raw_json = self.run(
            task=f"User request: {request}\n\nReturn the plan JSON.",
            context=context,
            json_mode=True,
        )
        parsed = json.loads(raw_json)
        steps = [
            PlanStep(order=int(step["order"]), agent=step["agent"], task=step["task"])
            for step in parsed.get("steps", [])
        ]
        steps.sort(key=lambda s: s.order)
        return steps

    def execute_plan(
        self,
        *,
        steps: Iterable[PlanStep],
        context: AgentContext,
        request: str,
    ) -> Dict[str, str]:
        outputs: Dict[str, str] = {}
        history: List[LLMMessage] = []
        for step in steps:
            agent = self.specialists.get(step.agent)
            if not agent:
                outputs[step.agent] = f"Agent `{step.agent}` not available."
                continue
            result = agent.run(task=step.task, context=context, history=history)
            outputs[step.agent] = result
            history.append({"role": "assistant", "content": result})
        return outputs

    def summarize(
        self,
        *,
        request: str,
        context: AgentContext,
        plan: Sequence[PlanStep],
        outputs: Dict[str, str],
    ) -> str:
        bullets = [
            f"Step {step.order} ({step.agent}): {outputs.get(step.agent, 'No output')}"
            for step in plan
        ]
        summary_context = AgentContext(
            dataset_description=context.dataset_description,
            project_goals=context.project_goals,
            constraints=context.constraints,
            additional_notes={"SpecialistOutputs": "\n".join(bullets)},
        )
        return self.run(
            task=f"User request: {request}\n\n{SUMMARY_PROMPT}",
            context=summary_context,
        )

    def orchestrate(self, request: str, context: AgentContext) -> Dict[str, str]:
        plan = self.create_plan(request, context)
        outputs = self.execute_plan(steps=plan, context=context, request=request)
        final_response = self.summarize(
            request=request, context=context, plan=plan, outputs=outputs
        )
        result: Dict[str, str] = {
            "final_response": final_response,
            "plan": json.dumps([step.__dict__ for step in plan], indent=2),
        }
        result.update({f"output_{name}": text for name, text in outputs.items()})
        return result


def build_default_orchestrator(client: OpenAIChatClient) -> GPT4Orchestrator:
    """Construct an orchestrator with the default specialist roster."""

    specialists = [
        LiteratureResearchAgent(client=client),
        DataCurationAgent(client=client),
        PropertyPredictionAgent(client=client),
        PolymerGenerationAgent(client=client),
        VisualizationAgent(client=client),
    ]
    return GPT4Orchestrator(client=client, specialists=specialists)
