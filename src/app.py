"""Entry point for running the polymer agent with GPT-4 orchestration."""
from __future__ import annotations

import os
from typing import Any, Dict

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .agent_tools import all_tools

SYSTEM_PROMPT = """
You are GPT-4 orchestrating a suite of polymer research tools. Follow this workflow:

1. Inspect the user's question and decide which tools are relevant.
2. Always prefer calling tools instead of relying on internal knowledge.
3. Combine evidence from multiple tools when necessary.
4. Return a concise final answer that cites the tools used.

Available tools:
- ``polymer_modality_processing_tool`` to build graph/geometry/fingerprint modalities from CSV data.
- ``contrastive_embedding_tool`` to obtain CL embeddings for processed polymers.
- ``polymer_property_prediction_tool`` to fit lightweight regressors over embeddings.
- ``polymer_generation_tool`` to retrieve similar polymers for ideation.
- ``polymer_guideline_search_tool`` to query the curated knowledge base.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def build_agent() -> AgentExecutor:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not configured")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    agent = create_tool_calling_agent(llm, all_tools, PROMPT)
    return AgentExecutor(agent=agent, tools=all_tools, verbose=True)


def run_agent_flow(user_request: str, context: str = "") -> str:
    """Run the orchestrator against a natural language instruction."""

    executor = build_agent()
    payload: Dict[str, Any] = {"input": f"CONTEXT: {context}\n\nREQUEST: {user_request}"}
    result = executor.invoke(payload)
    return result.get("output", "")


__all__ = ["build_agent", "run_agent_flow"]
