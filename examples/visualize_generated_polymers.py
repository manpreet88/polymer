"""Generate technical visuals for polymer SMILES candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from agents.visualization import PolymerVisualizationToolkit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create polymer visual reports")
    parser.add_argument(
        "--smiles",
        nargs="*",
        default=[],
        metavar="SMILES",
        help="Optional SMILES strings provided directly on the command line",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        default=Path("examples/sample_generated_smiles.json"),
        help="Path to a JSON list/object or text file containing SMILES (default: bundled sample)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visual_report"),
        help="Directory where visual assets will be written",
    )
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Disable histogram generation for the descriptor distribution",
    )
    return parser.parse_args()


def load_smiles_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"SMILES file not found: {path}")
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            candidate = data.get("smiles") or data.get("data")
        else:
            candidate = data
        if isinstance(candidate, str):
            return [candidate]
        if isinstance(candidate, list):
            return [str(item) for item in candidate]
        raise ValueError("JSON file must contain a list or an object with a 'smiles' key")
    return [line.strip() for line in text.splitlines() if line.strip()]


def combine_smiles(cli_smiles: Iterable[str], file_smiles: Iterable[str]) -> List[str]:
    smiles: List[str] = []
    for smi in cli_smiles:
        if smi:
            smiles.append(smi)
    for smi in file_smiles:
        if smi:
            smiles.append(smi)
    return smiles


def main() -> None:
    args = parse_args()
    smiles = combine_smiles(args.smiles, load_smiles_from_file(args.smiles_file))
    if not smiles:
        raise ValueError("No SMILES were provided via --smiles or --smiles-file")

    report = PolymerVisualizationToolkit.create_visual_report(
        smiles,
        output_dir=args.output_dir,
        include_histogram=not args.no_histogram,
    )

    print("Generated visual report:")
    print(f"- Molecule grid: {report.image_grid}")
    print(f"- Descriptor table: {report.descriptor_csv}")
    print(f"- Scatter plot: {report.scatter_plot}")
    if report.histogram_plot:
        print(f"- Histogram: {report.histogram_plot}")


if __name__ == "__main__":
    main()
