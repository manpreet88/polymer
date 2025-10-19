"""High level agent workflows inspired by the oncology decision-support paper."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agent_pipeline import (
    CheckpointConfig,
    KnowledgeBaseConfig,
    build_pipeline_services,
    run_expert_design_round,
    run_non_expert_walkthrough,
    seed_knowledge_base_from_catalog,
)

logging.basicConfig(level=logging.INFO)


def _print_block(title: str, payload) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print(json.dumps(payload, indent=2))


def main() -> None:
    checkpoint_cfg = CheckpointConfig().resolve()
    kb_cfg = KnowledgeBaseConfig().resolve()
    services = build_pipeline_services(checkpoint_cfg=checkpoint_cfg, knowledge_base_cfg=kb_cfg)

    catalog_path = Path(__file__).with_name("data").joinpath("polymer_reference.csv")
    seed_knowledge_base_from_catalog(services, catalog_path, overwrite=False)

    non_expert_payload = run_non_expert_walkthrough(
        services,
        psmiles="[*]OCF",
        property_focus=["glass_transition_c", "density_g_cm3"],
        neighbours=3,
    )
    _print_block("Non-expert triage summary", non_expert_payload)

    image_dir = Path(__file__).with_name("visualisations")
    expert_payload = run_expert_design_round(
        services,
        design_brief="Identify fluorinated polymers with improved thermal stability for sensor encapsulation.",
        seed_psmiles="[*]C(F)(F)C([*])F",
        candidate_count=3,
        mutations=2,
        image_dir=image_dir,
    )
    _print_block("Expert design round", expert_payload)


if __name__ == "__main__":
    main()
