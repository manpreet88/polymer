"""Executable casebook showcasing expert and non-expert polymer scenarios."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from agent_pipeline import (
    CheckpointConfig,
    KnowledgeBaseConfig,
    build_pipeline_services,
    seed_knowledge_base_from_catalog,
)
from agent_pipeline.casebook import CaseSpec, CasebookRunner


logging.basicConfig(level=logging.INFO)


def _default_cases() -> List[CaseSpec]:
    return [
        CaseSpec(
            title="Lab technician | Regrind-safe polycarbonate segment",
            persona="non_expert",
            psmiles="[*]OC(=O)C1CCC(CC1)OC(=O)[*]",
            objective="Summarise density and glass transition behaviour for a bisphenol-A polycarbonate repeat unit.",
            property_focus=["density_g_cm3", "glass_transition_c"],
            neighbours=5,
            evidence_query="bisphenol A polycarbonate Tg density safety datasheet",
        ),
        CaseSpec(
            title="Process engineer | Fluorinated barrier laminate",
            persona="expert",
            psmiles="[*]C(F)(F)C(F)(F)C([*])(F)F",
            objective="Identify fluorinated barrier polymers exceeding 150 °C Tg and suggest higher stability variants.",
            property_focus=["glass_transition_c", "thermal_stability_c"],
            include_generation=True,
            neighbours=6,
            evidence_query="fluorinated polymer barrier film thermal stability",
        ),
        CaseSpec(
            title="Sustainability analyst | Compostable packaging backbone",
            persona="non_expert",
            psmiles="[*]OC(=O)CC(=O)O[*]",
            objective="Explain biodegradation context and key thermal properties for an aliphatic polyester fragment.",
            property_focus=["decomposition_onset_c", "glass_transition_c"],
            neighbours=4,
            evidence_query="aliphatic polyester biodegradable packaging Tg decomposition temperature",
        ),
        CaseSpec(
            title="Additive manufacturing specialist | Elastomer blend tuning",
            persona="expert",
            psmiles="[*]CC=CC[*]",
            objective="Balance elasticity and solvent resistance; recommend co-monomers or fillers to stiffen selectively.",
            property_focus=["youngs_modulus_gpa", "swelling_index_percent"],
            include_generation=True,
            neighbours=4,
            evidence_query="polybutadiene solvent resistance additive manufacturing blend modifiers",
        ),
    ]


def main() -> None:
    checkpoint_cfg = CheckpointConfig().resolve()
    kb_cfg = KnowledgeBaseConfig().resolve()
    services = build_pipeline_services(checkpoint_cfg=checkpoint_cfg, knowledge_base_cfg=kb_cfg)

    catalog_path = Path(__file__).with_name("data").joinpath("polymer_reference.csv")
    if services.knowledge_base.is_empty() and catalog_path.exists():
        logging.info("Seeding knowledge base from %s", catalog_path)
        seed_knowledge_base_from_catalog(services, catalog_path, overwrite=False)

    runner = CasebookRunner(services=services)
    cases = _default_cases()
    outcomes = runner.run_all(cases)

    for outcome in outcomes:
        print("\n" + "=" * 100)
        print(outcome.spec.title)
        summary = {
            "persona": outcome.spec.persona,
            "objective": outcome.spec.objective,
            "input_psmiles": outcome.polymer_psmiles,
            "predictions": outcome.predictions,
            "neighbours": outcome.neighbours,
            "candidates": outcome.candidates,
            "evidence": outcome.evidence,
            "report_markdown": outcome.report_markdown,
            "report_figure": outcome.report_figure,
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
