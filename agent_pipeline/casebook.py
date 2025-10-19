"""Scenario runner that mirrors the oncology-agent casebook for polymers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .pipelines import PipelineServices, build_pipeline_services
from .reporting import ReportBuilder
from .tools.web_rag import WebRAGTool


@dataclass
class CaseSpec:
    """Configuration for a single agent-guided polymer analysis."""

    title: str
    persona: str  # "non_expert" or "expert"
    psmiles: str
    objective: str
    property_focus: Optional[Sequence[str]] = None
    include_generation: bool = False
    neighbours: int = 5
    evidence_query: Optional[str] = None
    extra_notes: Optional[str] = None


@dataclass
class CaseOutcome:
    """Structured result returned for each scenario."""

    spec: CaseSpec
    polymer_psmiles: str
    predictions: Dict[str, Dict[str, float]]
    neighbours: List[Dict[str, object]]
    candidates: List[Dict[str, object]]
    evidence: List[Dict[str, object]]
    report_markdown: str
    report_figure: Optional[str]
    tool_trace: List[Dict[str, object]] = field(default_factory=list)

    def to_serialisable(self) -> Dict[str, object]:
        """Return a JSON-ready payload."""
        return {
            "title": self.spec.title,
            "persona": self.spec.persona,
            "objective": self.spec.objective,
            "input_psmiles": self.polymer_psmiles,
            "predictions": self.predictions,
            "neighbours": self.neighbours,
            "candidates": self.candidates,
            "evidence": self.evidence,
            "report_markdown": self.report_markdown,
            "report_figure": self.report_figure,
            "tool_trace": self.tool_trace,
        }


class CasebookRunner:
    """High-level orchestrator that stages multiple polymer-agent scenarios."""

    def __init__(
        self,
        services: Optional[PipelineServices] = None,
        *,
        rag_tool: Optional[WebRAGTool] = None,
        reporter: Optional[ReportBuilder] = None,
    ) -> None:
        self.services = services or build_pipeline_services()
        self.rag = rag_tool or WebRAGTool()
        self.reporter = reporter or ReportBuilder()

    def _predict(self, psmiles: str, focus: Optional[Sequence[str]]) -> Dict[str, Dict[str, float]]:
        polymer = self.services.ingestion.ingest_psmiles(psmiles, source="casebook")
        return self.services.predictor.predict(polymer, properties=focus)

    def run_case(self, spec: CaseSpec) -> CaseOutcome:
        trace: List[Dict[str, object]] = []
        polymer = self.services.ingestion.ingest_psmiles(spec.psmiles, source=spec.persona)
        trace.append({"tool": "ingest_psmiles", "psmiles": polymer.psmiles})

        embedding = self.services.embedding.embed_polymer(polymer)
        trace.append({"tool": "embed_polymer", "vector_dim": len(embedding.vector)})

        predictions = self.services.predictor.predict_from_embedding(embedding, psmiles=polymer.psmiles)
        if spec.property_focus:
            requested = set(spec.property_focus)
            predictions = [pred for pred in predictions if pred.name in requested]
        prediction_payload: Dict[str, Dict[str, float]] = {}
        for pred in predictions:
            entry = {"mu": float(pred.mean), "sigma": float(pred.std)}
            if pred.unit is not None:
                entry["unit"] = pred.unit
            if pred.raw_outputs:
                entry["raw_outputs"] = {k: float(v) for k, v in pred.raw_outputs.items()}
            prediction_payload[pred.name] = entry
        trace.append({"tool": "predict_properties", "properties": list(prediction_payload)})

        neighbour_payload: List[Dict[str, object]] = []
        search_results = self.services.knowledge_base.search(embedding.vector, top_k=spec.neighbours)
        for entry, score in search_results:
            neighbour_payload.append(
                {
                    "psmiles": entry.psmiles,
                    "source": entry.source,
                    "similarity": float(score),
                    "name": entry.metadata.get("name"),
                    "metadata": entry.metadata,
                }
            )
        trace.append({"tool": "search_knowledge_base", "count": len(neighbour_payload)})

        candidate_payload: List[Dict[str, object]] = []
        if spec.include_generation:
            generated = self.services.generator.generate(
                polymer,
                top_k=max(spec.neighbours, 5),
                mutations=3,
            )
            for candidate in generated:
                try:
                    candidate_polymer = self.services.ingestion.ingest_psmiles(
                        candidate.psmiles, source="candidate", metadata=candidate.metadata
                    )
                    candidate_predictions = self.services.predictor.predict(candidate_polymer, properties=spec.property_focus)
                except Exception as exc:  # pragma: no cover - defensive guard
                    candidate_predictions = {"error": str(exc)}
                payload = {
                    "psmiles": candidate.psmiles,
                    "metadata": candidate.metadata,
                    "predictions": candidate_predictions,
                }
                candidate_payload.append(payload)
            trace.append({"tool": "generate_candidates", "count": len(candidate_payload)})

        query = spec.evidence_query or f"polymer {polymer.psmiles} applications properties safety"
        evidence = self.rag.tool_retrieve({"query": query, "top_k": 6})
        trace.append({"tool": "retrieve_context", "results": len(evidence), "query": query})

        report_markdown, figure_path = self.reporter.build(
            role=spec.persona,
            psmiles=polymer.psmiles,
            neighbors=neighbour_payload,
            predictions=prediction_payload,
            candidates=[
                {"psmiles": cand["psmiles"], "score": cand["metadata"].get("similarity")}
                for cand in candidate_payload
            ],
            evidence=evidence,
            extra_notes=spec.extra_notes or spec.objective,
        )
        trace.append({"tool": "write_report", "figure": figure_path})

        return CaseOutcome(
            spec=spec,
            polymer_psmiles=polymer.psmiles,
            predictions=prediction_payload,
            neighbours=neighbour_payload,
            candidates=candidate_payload,
            evidence=evidence,
            report_markdown=report_markdown,
            report_figure=figure_path,
            tool_trace=trace,
        )

    def run_all(self, cases: Sequence[CaseSpec]) -> List[CaseOutcome]:
        return [self.run_case(case) for case in cases]
