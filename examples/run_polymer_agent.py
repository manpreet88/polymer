"""Command line interface for the polymer AI agent system."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agents import (
    AgentContext,
    ModelConfig,
    OpenAIChatClient,
    build_default_orchestrator,
)
from agents.visualization import PolymerVisualizationToolkit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the polymer agent orchestrator")
    parser.add_argument("request", help="High level request for the orchestrator")
    parser.add_argument(
        "--dataset-description",
        default="Processed polymer modalities including SMILES, graphs, 3D geometries, and fingerprints.",
        help="Optional dataset description for context",
    )
    parser.add_argument(
        "--project-goals",
        default="Develop oncology-focused polymer candidates with interpretable evaluation.",
        help="High-level goals",
    )
    parser.add_argument(
        "--constraints",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Optional constraint key/value pairs",
    )
    parser.add_argument(
        "--notes",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional context notes",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Override orchestrator model (default: gpt-4o)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the orchestrator output JSON",
    )
    parser.add_argument(
        "--visualize-smiles",
        nargs="*",
        default=[],
        metavar="SMILES_OR_FILE",
        help=(
            "Optional SMILES strings or file paths to visualise after the orchestrator run. "
            "Files can be JSON (list or object with 'smiles') or plain text with one SMILES per line."
        ),
    )
    parser.add_argument(
        "--visual-output-dir",
        type=Path,
        default=Path("visual_reports"),
        help="Directory to store generated visual assets when --visualize-smiles is used",
    )
    return parser.parse_args()


def key_value_pairs(pairs: Any) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid key/value pair: {pair}")
        key, value = pair.split("=", 1)
        result[key] = value
    return result


def load_smiles_entries(entries: Iterable[str]) -> List[str]:
    smiles: List[str] = []
    for entry in entries:
        path = Path(entry)
        if path.exists():
            text = path.read_text().strip()
            if not text:
                continue
            if path.suffix.lower() == ".json":
                data = json.loads(text)
                if isinstance(data, dict):
                    candidate = data.get("smiles") or data.get("data")
                else:
                    candidate = data
                if isinstance(candidate, str):
                    smiles.append(candidate)
                elif isinstance(candidate, list):
                    smiles.extend(str(item) for item in candidate)
            else:
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped:
                        smiles.append(stripped)
        else:
            smiles.extend(s for s in entry.split(",") if s.strip())
    return smiles


def main() -> None:
    args = parse_args()

    constraints = key_value_pairs(args.constraints)
    notes = key_value_pairs(args.notes)

    client = OpenAIChatClient(default_model=args.model)
    orchestrator = build_default_orchestrator(client)

    # Allow overriding orchestrator model dynamically
    orchestrator.config.model_config = ModelConfig(model=args.model)

    context = AgentContext(
        dataset_description=args.dataset_description,
        project_goals=args.project_goals,
        constraints=constraints,
        additional_notes=notes,
    )

    result = orchestrator.orchestrate(args.request, context)
    output_str = json.dumps(result, indent=2)
    print(output_str)

    if args.output:
        args.output.write_text(output_str)
        print(f"Saved output to {args.output}")

    if args.visualize_smiles:
        smiles = load_smiles_entries(args.visualize_smiles)
        if not smiles:
            raise ValueError("--visualize-smiles was provided but no valid SMILES were found")
        report = PolymerVisualizationToolkit.create_visual_report(
            smiles, output_dir=args.visual_output_dir
        )
        print("\nGenerated visual assets:")
        print(f"- Molecule grid: {report.image_grid}")
        print(f"- Descriptor table: {report.descriptor_csv}")
        print(f"- Scatter plot: {report.scatter_plot}")
        if report.histogram_plot:
            print(f"- Histogram: {report.histogram_plot}")


if __name__ == "__main__":
    main()
