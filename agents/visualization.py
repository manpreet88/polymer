"""Utilities for visualising generated polymer candidates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

try:  # pragma: no cover - optional heavy dependency
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The `rdkit` package is required for polymer visualisations. Install it with "
        "`pip install rdkit-pypi`."
    ) from exc

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (requires Agg backend)


@dataclass
class VisualizationResult:
    """Container for generated artefacts."""

    smiles: List[str]
    descriptor_csv: Path
    image_grid: Path
    scatter_plot: Optional[Path] = None
    histogram_plot: Optional[Path] = None


class PolymerVisualizationToolkit:
    """Static helpers for converting SMILES into technical visuals."""

    DEFAULT_FIGSIZE = (6, 4)

    @staticmethod
    def _ensure_output_dir(output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def sanitise_smiles(smiles_list: Iterable[str]) -> List[str]:
        """Return a list of valid SMILES strings, filtering out invalid ones."""

        valid: List[str] = []
        for smiles in smiles_list:
            stripped = smiles.strip()
            if not stripped:
                continue
            mol = Chem.MolFromSmiles(stripped)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:  # pragma: no cover - RDKit raises specialised errors
                continue
            valid.append(Chem.MolToSmiles(mol))
        return valid

    @staticmethod
    def _mols_from_smiles(smiles: Sequence[str]) -> List[Chem.Mol]:
        mols: List[Chem.Mol] = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mols.append(mol)
        return mols

    @staticmethod
    def render_grid(
        smiles: Sequence[str],
        *,
        output_dir: Path,
        filename: str = "polymer_candidates.png",
        mols_per_row: int = 3,
        sub_img_size: tuple[int, int] = (300, 300),
        legends: Optional[Sequence[str]] = None,
    ) -> Path:
        """Render a grid image of the provided SMILES."""

        output_dir = PolymerVisualizationToolkit._ensure_output_dir(output_dir)
        mols = PolymerVisualizationToolkit._mols_from_smiles(smiles)
        if not mols:
            raise ValueError("No valid SMILES were provided for rendering.")

        if legends and len(legends) != len(mols):
            raise ValueError("Legend count must match the number of valid SMILES.")

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=list(legends) if legends else None,
        )
        output_path = output_dir / filename
        img.save(str(output_path))
        return output_path

    @staticmethod
    def compute_descriptors(smiles: Sequence[str]) -> List[Mapping[str, float]]:
        """Compute lightweight descriptors for each SMILES."""

        records: List[Mapping[str, float]] = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            records.append(
                {
                    "smiles": smi,
                    "mol_weight": Descriptors.MolWt(mol),
                    "logp": Descriptors.MolLogP(mol),
                    "tpsa": Descriptors.TPSA(mol),
                    "num_rings": float(Chem.GetSSSR(mol)),
                }
            )
        if not records:
            raise ValueError("No descriptors could be computed for the provided SMILES.")
        return records

    @staticmethod
    def save_descriptor_table(
        descriptors: Sequence[Mapping[str, float]], *, output_dir: Path, filename: str = "descriptors.csv"
    ) -> Path:
        output_dir = PolymerVisualizationToolkit._ensure_output_dir(output_dir)
        lines = ["smiles,mol_weight,logp,tpsa,num_rings"]
        for record in descriptors:
            lines.append(
                ",".join(
                    [
                        record["smiles"],
                        f"{record['mol_weight']:.4f}",
                        f"{record['logp']:.4f}",
                        f"{record['tpsa']:.4f}",
                        f"{record['num_rings']:.0f}",
                    ]
                )
            )
        output_path = output_dir / filename
        output_path.write_text("\n".join(lines))
        return output_path

    @staticmethod
    def plot_scatter(
        descriptors: Sequence[Mapping[str, float]],
        *,
        x_key: str = "mol_weight",
        y_key: str = "logp",
        output_dir: Path,
        filename: str = "descriptor_scatter.png",
    ) -> Path:
        output_dir = PolymerVisualizationToolkit._ensure_output_dir(output_dir)
        x = [record[x_key] for record in descriptors]
        y = [record[y_key] for record in descriptors]
        smiles = [record["smiles"] for record in descriptors]

        fig, ax = plt.subplots(figsize=PolymerVisualizationToolkit.DEFAULT_FIGSIZE)
        ax.scatter(x, y, c="#1f77b4", s=60, edgecolors="k", linewidths=0.5)
        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel(y_key.replace("_", " ").title())
        ax.set_title("Descriptor Scatter Plot")
        for sx, sy, label in zip(x, y, smiles):
            ax.annotate(label, (sx, sy), textcoords="offset points", xytext=(4, 4), fontsize=7)
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    @staticmethod
    def plot_histogram(
        descriptors: Sequence[Mapping[str, float]],
        *,
        key: str = "mol_weight",
        output_dir: Path,
        filename: str = "descriptor_histogram.png",
        bins: int = 10,
    ) -> Path:
        output_dir = PolymerVisualizationToolkit._ensure_output_dir(output_dir)
        values = [record[key] for record in descriptors]

        fig, ax = plt.subplots(figsize=PolymerVisualizationToolkit.DEFAULT_FIGSIZE)
        ax.hist(values, bins=bins, color="#ff7f0e", edgecolor="black")
        ax.set_xlabel(key.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {key.replace('_', ' ').title()}")
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    @staticmethod
    def create_visual_report(
        smiles: Iterable[str],
        *,
        output_dir: Path,
        include_histogram: bool = True,
    ) -> VisualizationResult:
        """Generate a bundle of visuals and descriptor table for the SMILES list."""

        valid_smiles = PolymerVisualizationToolkit.sanitise_smiles(smiles)
        if not valid_smiles:
            raise ValueError("No valid SMILES were provided for the visual report.")

        descriptors = PolymerVisualizationToolkit.compute_descriptors(valid_smiles)
        descriptor_csv = PolymerVisualizationToolkit.save_descriptor_table(
            descriptors, output_dir=output_dir
        )
        grid = PolymerVisualizationToolkit.render_grid(valid_smiles, output_dir=output_dir)
        scatter = PolymerVisualizationToolkit.plot_scatter(
            descriptors, output_dir=output_dir
        )
        histogram_path: Optional[Path] = None
        if include_histogram:
            histogram_path = PolymerVisualizationToolkit.plot_histogram(
                descriptors, output_dir=output_dir
            )

        return VisualizationResult(
            smiles=valid_smiles,
            descriptor_csv=descriptor_csv,
            image_grid=grid,
            scatter_plot=scatter,
            histogram_plot=histogram_path,
        )


__all__ = ["PolymerVisualizationToolkit", "VisualizationResult"]
