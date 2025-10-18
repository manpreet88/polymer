"""Example of an expert-oriented batch evaluation loop."""

from __future__ import annotations

import random
from typing import List

from polymer_agent import (
    AgentConfig,
    AgentMode,
    HybridRetriever,
    KnowledgeBaseConfig,
    PolymerAgentOrchestrator,
    PolymerKnowledgeBase,
)
from polymer_agent.visualization import format_table, TableColumn


RANDOM_SEED = 7


def seeded_embeddings(dim: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def build_expert_kb() -> PolymerKnowledgeBase:
    config = KnowledgeBaseConfig(autoreload=False)
    kb = PolymerKnowledgeBase(config)
    kb.add_polymer(
        "PEEK",
        tag="high_temp",
        embedding=seeded_embeddings(6, 1),
        processing_window="360-400C",
        citation="doi:10.1000/peek",
    )
    kb.add_polymer(
        "PEI",
        tag="high_temp",
        embedding=seeded_embeddings(6, 2),
        processing_window="340-380C",
        citation="doi:10.1000/pei",
    )
    kb.add_experiment(
        "Exp-001",
        tag="high_temp",
        property="Tg",
        value=215,
        uncertainty=3.5,
    )
    kb.add_edge("PEEK", "Exp-001", weight=0.6)
    kb.add_edge("PEI", "Exp-001", weight=0.7)
    return kb


def main() -> None:
    random.seed(RANDOM_SEED)
    config = AgentConfig()
    kb = build_expert_kb()
    retriever = HybridRetriever(kb, config.knowledge_base)
    orchestrator = PolymerAgentOrchestrator(config, kb, retriever)

    queries = [
        "heat resistant aerospace polymer",
        "high_temp",
    ]
    for query in queries:
        print(f"=== Query: {query} ===")
        response = orchestrator.handle_query(query, AgentMode.EXPERT)
        print(response)
        print()

    print("=== Knowledge Base Snapshot ===")
    polymer_nodes = list(kb.iter_nodes("polymer"))
    table = format_table(
        [
            TableColumn("Identifier", [node.identifier for node in polymer_nodes]),
            TableColumn(
                "Processing Window",
                [node.attributes.get("processing_window", "-") for node in polymer_nodes],
            ),
        ]
    )
    print(table)


if __name__ == "__main__":
    main()
