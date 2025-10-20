"""Entry point for running the polymer agent with GPT-4 orchestration (robust)."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Import tools with a shim so it works as script or module
try:
    from .agent_tools import all_tools
except Exception:
    from src.agent_tools import all_tools  # requires running from project root

DEBUG = os.environ.get("POLYMER_AGENT_DEBUG", "0") == "1"

SYSTEM_PROMPT = """
You are GPT-4 orchestrating a suite of polymer research tools. Follow this workflow:

1. Inspect the user's question and decide which tools are relevant.
2. Always prefer calling tools instead of relying on internal knowledge.
3. Combine evidence from multiple tools when necessary.
4. Return a concise final answer that cites the tools used.

Available tools:
- polymer_modality_processing_tool: build graph/geometry/fingerprint modalities from CSV data.
- contrastive_embedding_tool: obtain CL embeddings for processed polymers.
- polymer_property_prediction_tool: fit lightweight regressors over embeddings.
- polymer_generation_tool: retrieve similar polymers for ideation.
- polymer_guideline_search_tool: query the curated knowledge base.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

def _dprint(*args):
    if DEBUG:
        print("[app]", *args, file=sys.stderr)

def _extract_text(result: Any) -> str:
    """Handle many possible return shapes from different LC agent executors."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # Common keys across versions / wrappers:
        for key in ("output", "output_text", "final_output", "result"):
            if key in result and isinstance(result[key], str) and result[key].strip():
                return result[key]
        # LangChain Runnable-style: {"messages": [...]} etc.
        if "messages" in result and isinstance(result["messages"], list):
            try:
                return result["messages"][-1].content
            except Exception:
                pass
        # Last resort stringify
        return str(result)
    # RunnableOutput / BaseMessage, etc.
    try:
        text = getattr(result, "content", None)
        if isinstance(text, str) and text.strip():
            return text
    except Exception:
        pass
    return str(result)

def _build_agent_new_api(llm):
    """LangChain ≥0.2 style."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent  # type: ignore
    _dprint("Using NEW agent API (AgentExecutor + create_tool_calling_agent)")
    agent = create_tool_calling_agent(llm, all_tools, PROMPT)
    return AgentExecutor(agent=agent, tools=all_tools, verbose=True)

def _build_agent_legacy_api(llm):
    """Older LangChain style."""
    from langchain.agents import initialize_agent, AgentType  # type: ignore
    _dprint("Using LEGACY agent API (initialize_agent)")
    try:
        return initialize_agent(
            tools=all_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={"system_message": SYSTEM_PROMPT},
        )
    except TypeError:
        # Some versions don't accept agent_kwargs
        return initialize_agent(
            tools=all_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

def build_agent():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not configured")

    # Model name: if your env is older, gpt-4o-mini might be safer.
    model_name = os.environ.get("POLYMER_AGENT_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model_name, temperature=0.0)
    _dprint(f"LLM model: {model_name}")
    _dprint(f"Loaded {len(all_tools)} tools.")

    # Try new API
    try:
        return _build_agent_new_api(llm)
    except Exception as e_new:
        _dprint(f"NEW API unavailable: {e_new!r}")

    # Try legacy API
    try:
        return _build_agent_legacy_api(llm)
    except Exception as e_old:
        _dprint(f"LEGACY API unavailable: {e_old!r}")

    # As a safety net, return just the LLM (no tools)
    _dprint("Falling back to direct LLM (no agent).")
    class _BareLLM:
        def __init__(self, _llm):
            self._llm = _llm
        def invoke(self, payload: Dict[str, Any]):
            # Expect payload["input"]
            prompt = payload.get("input", "")
            msg = PROMPT.format_messages(input=prompt)
            return self._llm.invoke(msg)
    return _BareLLM(llm)

def run_agent_flow(user_request: str, context: str = "") -> str:
    """Run the orchestrator against a natural language instruction."""
    executor = build_agent()
    payload: Dict[str, Any] = {"input": f"CONTEXT: {context}\n\nREQUEST: {user_request}"}

    _dprint("Invoking agent/executor...")
    try:
        result = executor.invoke(payload)
    except AttributeError:
        # Some legacy executors use .run(...)
        try:
            result = executor.run(payload)
        except Exception as e:
            raise RuntimeError(f"Agent execution failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Agent execution failed: {e}") from e

    text = _extract_text(result)
    if not text.strip():
        _dprint(f"Raw result had no text. Raw: {result!r}")
        text = "(No text returned by agent/LLM.)"
    return text

__all__ = ["build_agent", "run_agent_flow"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the polymer agent")
    parser.add_argument(
        "-i", "--input", type=str, default=None,
        help="User request to send to the agent (if omitted, read from stdin)"
    )
    parser.add_argument(
        "-c", "--context", type=str, default="",
        help="Optional context string passed to the agent"
    )
    args = parser.parse_args()

    if args.input is None:
        data = sys.stdin.read().strip()
        if not data:
            print("No input provided. Use --input 'your question' or pipe text on stdin.", file=sys.stderr)
            sys.exit(1)
        args.input = data

    try:
        output = run_agent_flow(user_request=args.input, context=args.context)
        # Always print something
        print(output)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
