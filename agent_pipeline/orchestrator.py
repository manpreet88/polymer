# agent_pipeline/orchestrator.py
"""
GPT-4 Orchestrator with tool routing (no hardcoding).

Exposes:
  - embed_polymer(psmiles)
  - predict_properties(psmiles, properties?, use_kb?)  # runs your heads
  - generate_candidates(psmiles, top_k?, mutations?)
  - search_web(query)
  - retrieve_context(query, query_expansion?, top_k?)
  - write_report(role, psmiles, include_generation?, target_properties?)

Requires that the existing project provides:
  - PolymerDataIngestionTool
  - MultimodalEmbeddingService
  - PolymerKnowledgeBase (optional if you want persistence)
  - PropertyPredictorEnsemble
  - PolymerGeneratorTool

Env:
  OPENAI_API_KEY (and optional OPENAI_MODEL, OPENAI_ORG)
"""

from __future__ import annotations
import os, json, logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from .data_ingestion import PolymerDataIngestionTool
from .embedding_service import MultimodalEmbeddingService
from .knowledge_base import PolymerKnowledgeBase
from .property_predictor import PropertyPredictorEnsemble
from .generation import PolymerGeneratorTool
from .config import CheckpointConfig  # assumed present in your repo
from .datamodels import ProcessedPolymer  # assumed present in your repo

from .reporting import ReportBuilder
from .tools.web_rag import WebRAGTool

LOGGER = logging.getLogger(__name__)

# -------- Tool registry

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, description: str, func, parameters: Dict[str, Any]):
        self._tools[name] = {
            "name": name,
            "description": description,
            "func": func,
            "parameters": parameters,
        }

    def list(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, arguments_json: str):
        data = json.loads(arguments_json or "{}")
        return self._tools[name]["func"](data)

# -------- Orchestrator

class GPT4Orchestrator:
    def __init__(
        self,
        registry: ToolRegistry,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.registry = registry
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are a polymer science assistant. Choose and call tools to answer questions. "
            "Prefer evidence-grounded answers with retrieved context when helpful. "
            "Never fabricate values; when uncertain, ask for more info or call a tool."
        )
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.org = os.getenv("OPENAI_ORG")

    def chat(self, user_message: str) -> str:
        """
        Single-turn helper to show function-calling orchestration.
        """
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.org:
            headers["OpenAI-Organization"] = self.org

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "tools": self.registry.list(),
            "tool_choice": "auto",
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        js = r.json()
        msg = js["choices"][0]["message"]

        if "tool_calls" not in msg:
            return msg.get("content", "")

        # Execute each tool call and send back results
        tool_outputs = []
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = call["function"].get("arguments", "{}")
            result = self.registry.call(name, args)
            tool_outputs.append({"tool_call_id": call["id"], "name": name, "result": result})

        # Follow-up turn with tool outputs
        follow_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
                msg,
                *[
                    {
                        "role": "tool",
                        "tool_call_id": t["tool_call_id"],
                        "name": t["name"],
                        "content": json.dumps(t["result"]),
                    }
                    for t in tool_outputs
                ],
            ],
        }
        r2 = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=follow_payload, timeout=120)
        r2.raise_for_status()
        return r2.json()["choices"][0]["message"].get("content", "")

# -------- Wiring convenience

def build_default_services() -> Dict[str, Any]:
    ckpt = CheckpointConfig().resolve()
    ingestion = PolymerDataIngestionTool()
    embedding = MultimodalEmbeddingService(ckpt)
    # Knowledge base is optional; if present, it can persist vectors (your repo already supports it)
    try:
        kb = PolymerKnowledgeBase.from_default().load_if_exists()
    except Exception:
        kb = None
    predictor = PropertyPredictorEnsemble(ckpt, embedding_service=embedding)
    generator = PolymerGeneratorTool(embedding, kb, predictor)
    rag = WebRAGTool()
    reporter = ReportBuilder()
    return dict(ingestion=ingestion, embedding=embedding, kb=kb, predictor=predictor, generator=generator, rag=rag, reporter=reporter)

def register_default_tools(registry: ToolRegistry, sv: Dict[str, Any]) -> None:
    ingestion, embedding, kb, predictor, generator, rag, reporter = (
        sv["ingestion"], sv["embedding"], sv["kb"], sv["predictor"], sv["generator"], sv["rag"], sv["reporter"]
    )

    # Embed
    registry.register(
        "embed_polymer",
        "Embed a polymer from P-SMILES using multimodal encoder",
        lambda a: embedding.embed_polymer(ingestion.ingest_psmiles(a["psmiles"])).vector.tolist(),
        {"type": "object", "properties": {"psmiles": {"type": "string"}}, "required": ["psmiles"]},
    )

    # Predict
    def _predict(a):
        p = ingestion.ingest_psmiles(a["psmiles"])
        preds = predictor.predict(p, properties=a.get("properties"))
        # Ensure JSON serializable (float)
        out = {k: {"mu": float(v["mu"]), "sigma": float(v.get("sigma", 0.0))} for k, v in preds.items()}
        return out

    registry.register(
        "predict_properties",
        "Predict polymer properties; returns dict of {property: {mu, sigma}}",
        _predict,
        {
            "type": "object",
            "properties": {
                "psmiles": {"type": "string"},
                "properties": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["psmiles"],
        },
    )

    # Generate candidates
    registry.register(
        "generate_candidates",
        "Generate candidate polymers via retrieval + mutation (uses your generator)",
        lambda a: [
            {"psmiles": c.psmiles, **({ "score": float(getattr(c, "score", 0.0)) } if hasattr(c, "score") else {})}
            for c in generator.generate(
                ingestion.ingest_psmiles(a["psmiles"]),
                top_k=int(a.get("top_k", 5)),
                mutations=int(a.get("mutations", 2)),
            )
        ],
        {
            "type": "object",
            "properties": {"psmiles": {"type": "string"}, "top_k": {"type": "integer"}, "mutations": {"type": "integer"}},
            "required": ["psmiles"],
        },
    )

    # Optional KB search if user has persisted KB
    if kb is not None:
        registry.register(
            "search_knowledge_base",
            "Search persisted polymer KB by vector similarity",
            lambda a: [
                {"psmiles": e.psmiles, "similarity": float(s)}
                for e, s in kb.search(a["vector"], top_k=int(a.get("top_k", 5)))
            ],
            {
                "type": "object",
                "properties": {
                    "vector": {"type": "array", "items": {"type": "number"}},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["vector"],
            },
        )

    # Web RAG
    registry.register(
        "search_web",
        "Web search for polymer background (uses Tavily/SerpAPI)",
        lambda a: rag.tool_search({"query": a["query"]}),
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    registry.register(
        "retrieve_context",
        "Retrieve top-ranked evidence snippets from the web for a query",
        lambda a: rag.tool_retrieve({"query": a["query"], "query_expansion": a.get("query_expansion"), "top_k": a.get("top_k", 6)}),
        {
            "type": "object",
            "properties": {"query": {"type": "string"}, "query_expansion": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    )

    # Report
    def _write_report(a):
        psmiles = a["psmiles"]
        role = a.get("role", "non-expert")
        inc_gen = bool(a.get("include_generation", False))

        # Core computations
        p = ingestion.ingest_psmiles(psmiles)
        vec = embedding.embed_polymer(p).vector.tolist()

        neighbors = []
        if kb is not None:
            neighbors = [{"psmiles": e.psmiles, "similarity": float(s)} for e, s in kb.search(vec, top_k=5)]

        preds = predictor.predict(p)
        cands = []
        if inc_gen:
            cands = [
                {"psmiles": c.psmiles, **({"score": float(getattr(c, "score", 0.0))} if hasattr(c, "score") else {})}
                for c in generator.generate(p, top_k=5, mutations=2)
            ]

        # Evidence
        query = a.get("evidence_query") or f"polymer {psmiles} properties processing safety applications"
        evidence = rag.tool_retrieve({"query": query, "top_k": 6})

        md, fig_path = reporter.build(role, psmiles, neighbors, preds, cands, evidence, a.get("notes"))
        return {"markdown": md, "figure_path": fig_path}

    registry.register(
        "write_report",
        "Build a role-aware Markdown report with visuals and web evidence",
        _write_report,
        {
            "type": "object",
            "properties": {
                "role": {"type": "string", "enum": ["non-expert", "expert"]},
                "psmiles": {"type": "string"},
                "include_generation": {"type": "boolean"},
                "evidence_query": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["psmiles"],
        },
    )
