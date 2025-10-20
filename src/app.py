"""Entry point for running the polymer agent with GPT-4 orchestration (robust)."""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Callable, Optional

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover - optional dependency
    ChatPromptTemplate = None  # type: ignore

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

if ChatPromptTemplate is not None:
    PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
else:
    PROMPT = None

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

class _LocalToolAgent:
    """Fallback executor that runs tools directly when no LLM is available."""

    def __init__(self):
        self._tools = {tool.name: tool for tool in all_tools}

    @staticmethod
    def _strip_context(request: str) -> str:
        if "REQUEST:" in request:
            return request.split("REQUEST:", 1)[1].strip()
        return request.strip()

    @staticmethod
    def _extract_number(text: str, *keywords: str, default: Optional[float] = None) -> Optional[float]:
        for keyword in keywords:
            pattern = rf"{keyword}[\s:=]+([0-9]+(?:\.[0-9]+)?)"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                except ValueError:
                    continue
                return value
        return default

    @staticmethod
    def _extract_path(text: str, extension: str, occurrence: int = 0) -> Optional[str]:
        pattern = rf"([\w./\\-]+{re.escape(extension)})"
        matches = re.findall(pattern, text)
        if len(matches) > occurrence:
            return matches[occurrence]
        return None

    @staticmethod
    def _extract_quoted(text: str, *keywords: str) -> Optional[str]:
        if keywords:
            keyword_pattern = "|".join(re.escape(k) for k in keywords)
            scoped = re.search(rf"(?:{keyword_pattern}).*?['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
            if scoped:
                return scoped.group(1)
        quotes = re.findall(r"['\"]([^'\"]+)['\"]", text)
        return quotes[0] if quotes else None

    @staticmethod
    def _call_tool(tool, **kwargs) -> Any:
        func: Callable[..., Any]
        func = getattr(tool, "func", None) or getattr(tool, "run", None)
        if func is None:
            raise RuntimeError(f"Tool {getattr(tool, 'name', tool)} is not callable")
        return func(**kwargs) if kwargs else func()

    def _handle_modality(self, request: str) -> str:
        csv_path = self._extract_path(request, ".csv")
        if not csv_path:
            raise ValueError("polymer_modality_processing_tool requires a .csv path")
        chunk = self._extract_number(request, "chunk_size", "chunk") or 500
        workers = self._extract_number(request, "num_workers", "workers") or 4
        result = self._call_tool(
            self._tools["polymer_modality_processing_tool"],
            csv_path=csv_path,
            chunk_size=int(chunk),
            num_workers=int(workers),
        )
        return f"polymer_modality_processing_tool output: {result}"

    def _handle_contrastive(self, request: str) -> str:
        samples_path = self._extract_path(request, ".jsonl")
        if not samples_path:
            raise ValueError("contrastive_embedding_tool requires a .jsonl samples file")
        result = self._call_tool(
            self._tools["contrastive_embedding_tool"],
            samples_path=samples_path,
        )
        return json.dumps(result, indent=2)

    def _handle_property_prediction(self, request: str) -> str:
        dataset = self._extract_path(request, ".jsonl")
        if not dataset:
            raise ValueError("polymer_property_prediction_tool requires a .jsonl dataset")
        target = self._extract_quoted(request, "target", "target_key")
        if not target:
            raise ValueError("Specify the target key using quotes, e.g. 'target_key \"Tg\"'")
        train_fraction = self._extract_number(request, "train_fraction", "train split") or 0.8
        result = self._call_tool(
            self._tools["polymer_property_prediction_tool"],
            dataset_path=dataset,
            target_key=target,
            train_fraction=float(train_fraction),
        )
        return json.dumps(result, indent=2)

    def _handle_generation(self, request: str) -> str:
        seed = self._extract_path(request, ".jsonl", occurrence=0)
        library = self._extract_path(request, ".jsonl", occurrence=1)
        if not seed or not library:
            raise ValueError("polymer_generation_tool requires seed and library .jsonl paths")
        top_k = self._extract_number(request, "top_k", "top") or 5
        result = self._call_tool(
            self._tools["polymer_generation_tool"],
            seed_path=seed,
            library_path=library,
            top_k=int(top_k),
        )
        return json.dumps(result, indent=2)

    def _handle_guideline(self, request: str) -> str:
        after_name = re.split(r"polymer_guideline_search_tool", request, flags=re.IGNORECASE)
        query = after_name[1] if len(after_name) > 1 else request
        query = query.strip()
        query = re.sub(r"^(to|for|about)\s+", "", query, flags=re.IGNORECASE)
        if not query:
            raise ValueError("polymer_guideline_search_tool requires a search query")
        return self._call_tool(
            self._tools["polymer_guideline_search_tool"],
            query=query,
        )

    def _dispatch(self, request: str) -> str:
        if "polymer_modality_processing_tool" in request:
            return self._handle_modality(request)
        if "contrastive_embedding_tool" in request:
            return self._handle_contrastive(request)
        if "polymer_property_prediction_tool" in request:
            return self._handle_property_prediction(request)
        if "polymer_generation_tool" in request:
            return self._handle_generation(request)
        if "polymer_guideline_search_tool" in request:
            return self._handle_guideline(request)
        raise ValueError("Could not determine which tool to run from the request")

    def invoke(self, payload: Dict[str, Any]) -> str:
        request = str(payload.get("input", ""))
        clean_request = self._strip_context(request)
        try:
            return self._dispatch(clean_request)
        except Exception as exc:
            return f"Tool execution failed: {exc}"


def build_agent():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        _dprint("OPENAI_API_KEY not set; using local tool executor.")
        return _LocalToolAgent()
    if ChatOpenAI is None or ChatPromptTemplate is None or PROMPT is None:
        _dprint("LangChain dependencies missing; using local tool executor.")
        return _LocalToolAgent()

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
