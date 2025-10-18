"""Example script demonstrating ingestion, retrieval and question answering."""
from __future__ import annotations

from agent import AgentConfig, PolymerRAGAgent


EXAMPLE_POLYMERS = {
    "PEG": {
        "psmiles": "OCCOCCOCCO",
        "metadata": {"name": "Polyethylene glycol", "notes": "Hydrophilic polymer."},
    },
    "PTFE": {
        "psmiles": "FC(F)(F)C(F)(F)F",
        "metadata": {"name": "Polytetrafluoroethylene", "notes": "Highly inert."},
    },
}


def main() -> None:
    config = AgentConfig()
    agent = PolymerRAGAgent(config=config)

    for identifier, payload in EXAMPLE_POLYMERS.items():
        agent.ingest(identifier, payload["psmiles"], payload.get("metadata"))

    response = agent.answer(
        "Compare the flexibility of PEG and PTFE and suggest an application for each.",
        query_psmiles="OCCOCCOCCO",
    )
    print("Answer:\n", response["answer"])
    print("\nRetrieved entries:")
    for block in response["retrieved"]:
        print("-" * 40)
        print(block)


if __name__ == "__main__":
    main()
