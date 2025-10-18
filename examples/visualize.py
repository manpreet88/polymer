"""Generate 2-D and 3-D visualisations for a polymer."""
from __future__ import annotations

from pathlib import Path

from agent import PolymerPreprocessor, render_2d, render_3d


def main() -> None:
    preprocessor = PolymerPreprocessor()
    sample = preprocessor.process("OCCOCCOCCO")
    output_dir = Path("visualizations")
    png_path = render_2d(sample.canonical_psmiles, output_dir / "peg.png")
    try:
        html_path = render_3d(sample.canonical_psmiles, output_dir / "peg.html")
    except RuntimeError:
        html_path = None
    print(f"2-D depiction stored at: {png_path}")
    if html_path:
        print(f"Interactive 3-D visualisation stored at: {html_path}")
    else:
        print("Plotly not installed; skipped 3-D visualisation.")


if __name__ == "__main__":
    main()
