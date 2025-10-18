"""Show how to evaluate a multimodal bundle using the orchestrator."""

from __future__ import annotations

from polymer_agent import (
    AgentConfig,
    AgentMode,
    HybridRetriever,
    KnowledgeBaseConfig,
    PolymerAgentOrchestrator,
    PolymerKnowledgeBase,
)
from polymer_agent.data import MultimodalDataBundle


def build_stub_bundle() -> MultimodalDataBundle:
    return MultimodalDataBundle(
        psmiles="C[C@H](O)C(=O)O",
        graph={"nodes": 42, "edges": 44},
        geometry={"conformers": 1},
        fingerprint={"bits": [0, 1, 0, 1]},
        metadata={"source": "demo", "embedding": [0.05, 0.1, 0.15]},
    )


def build_kb() -> PolymerKnowledgeBase:
    config = KnowledgeBaseConfig(autoreload=False)
    kb = PolymerKnowledgeBase(config)
    kb.add_polymer(
        "PLA",
        tag="biodegradable",
        embedding=[0.04, 0.09, 0.16],
        references=["doi:10.1000/example"],
    )
    return kb


def main() -> None:
    config = AgentConfig()
    kb = build_kb()
    retriever = HybridRetriever(kb, config.knowledge_base)
    orchestrator = PolymerAgentOrchestrator(config, kb, retriever)

    bundle = build_stub_bundle()
    orchestrator.handle_query("biodegradable", AgentMode.NON_EXPERT)
    report = orchestrator.evaluate_candidate(bundle, AgentMode.NON_EXPERT)
    print(report)


if __name__ == "__main__":
    main()
