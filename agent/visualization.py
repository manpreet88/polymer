"""Visualization helpers for polymer structures."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    go = None


def render_2d(psmiles: str, output_path: Path, size: int = 400) -> Path:
    """Create a 2-D depiction of the polymer and save it as a PNG."""

    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        raise ValueError("Unable to parse PSMILES for rendering")
    AllChem.Compute2DCoords(mol)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Draw.MolToFile(mol, str(output_path), size=(size, size))
    return output_path


def render_3d(psmiles: str, output_path: Optional[Path] = None) -> Optional[Path]:
    """Create a simple 3-D scatter depiction using Plotly if available."""

    if go is None:
        raise RuntimeError("Plotly is required for 3-D rendering. Install plotly>=5.0.")
    mol = Chem.AddHs(Chem.MolFromSmiles(psmiles))
    if mol is None:
        raise ValueError("Unable to parse PSMILES for rendering")
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    coords = np.array(conf.GetPositions())
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(size=6, color=np.arange(len(symbols)), colorscale="Viridis"),
                text=symbols,
            )
        ]
    )
    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        return output_path
    return None


__all__ = ["render_2d", "render_3d"]
