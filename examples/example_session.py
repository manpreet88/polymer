# examples/high_level_cases.py
"""
Run high-level scenarios without hardcoding the orchestrator flow.

Usage:
  python -m examples.high_level_cases --role non-expert --psmiles "[*]CCO"
  python -m examples.high_level_cases --role expert --psmiles "[*]CCO" --include-generation

Requires env:
  OPENAI_API_KEY (and OPENAI_MODEL optional)
  TAVILY_API_KEY or SERPAPI_API_KEY for web RAG
"""

from __future__ import annotations
import argparse, json
from agent_pipeline.orchestrator import ToolRegistry, GPT4Orchestrator, build_default_services, register_default_tools

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="non-expert", choices=["non-expert", "expert"])
    ap.add_argument("--psmiles", required=True, help="Polymer P-SMILES, e.g., \"[*]CCO\"")
    ap.add_argument("--include-generation", action="store_true")
    args = ap.parse_args()

    sv = build_default_services()
    reg = ToolRegistry()
    register_default_tools(reg, sv)
    agent = GPT4Orchestrator(registry=reg)

    # Let GPT-4 decide, but give it a helpful instruction message
    prompt = (
        f"A user asks about the polymer {args.psmiles}.\n"
        f"Role = {args.role}.\n"
        f"1) Embed and predict likely properties.\n"
        f"2) Retrieve web context.\n"
        f"3) "
        + ("Also generate candidates and include them in the report.\n" if args.include_generation else "Skip generation for now.\n")
        + "4) Use write_report to produce the final answer."
    )

    out = agent.chat(prompt)
    print(out)

if __name__ == "__main__":
    main()
