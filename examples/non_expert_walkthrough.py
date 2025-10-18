"""Demonstration script showing a non-expert conversation flow."""

from __future__ import annotations

from pathlib import Path

from polymer_agent import (
    AgentConfig,
    AgentMode,
    HybridRetriever,
    KnowledgeBaseConfig,
    ModelRegistryPaths,
    PolymerAgentOrchestrator,
    PolymerKnowledgeBase,
)


def build_in_memory_kb() -> PolymerKnowledgeBase:
    config = KnowledgeBaseConfig(autoreload=False)
    kb = PolymerKnowledgeBase(config)
    kb.add_polymer(
        "PLA",
        tag="biodegradable",
        description="Polylactic acid with moderate strength",
        embedding=[0.1, 0.2, 0.3],
        density=1.24,
    )
    kb.add_polymer(
        "PBAT",
        tag="flexible",
        description="Polybutylene adipate terephthalate",
        embedding=[0.3, 0.1, 0.2],
        density=1.23,
    )
    kb.add_reference(
        "doi:10.1000/example",
        tag="biodegradable",
        summary="Review of compostable packaging polymers",
    )
    kb.add_edge("PLA", "doi:10.1000/example", weight=0.8)
    return kb


def main() -> None:
    config = AgentConfig()
    kb = build_in_memory_kb()
    retriever = HybridRetriever(kb, config.knowledge_base)
    orchestrator = PolymerAgentOrchestrator(config, kb, retriever)

    print("=== Conversation ===")
    response = orchestrator.handle_query("biodegradable cold-chain film", AgentMode.NON_EXPERT)
    print(response)


if __name__ == "__main__":
    main()
