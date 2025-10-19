# agent_pipeline/reporting.py
"""
ReportBuilder: creates role-aware markdown reports with a small matplotlib visual.

- Non-expert: simpler language, definitions, applications, safety.
- Expert: technical tone, neighbors, μ±σ, constraints, and citations.

Produces:
  - Markdown string
  - Saves a PNG bar chart of predicted properties with μ±σ.

This module does not fetch; it just formats provided results.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import datetime as dt
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

def _fmt_mu_sigma(mu: float, sigma: float) -> str:
    return f"{mu:.3g} ± {sigma:.2g}"

def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

class ReportBuilder:
    def __init__(self, out_dir: str = "reports"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _plot_properties(self, predictions: Dict[str, Dict[str, float]], image_name: str) -> str:
        props, mus, sigs = [], [], []
        for k, v in predictions.items():
            mu, sigma = float(v.get("mu", 0.0)), float(v.get("sigma", 0.0))
            props.append(k)
            mus.append(mu)
            sigs.append(sigma)
        if not props:
            return ""
        plt.figure(figsize=(6.5, 4.0))
        x = range(len(props))
        plt.bar(x, mus, yerr=sigs, capsize=4)   # no explicit colors per style rule
        plt.xticks(list(x), props, rotation=20, ha="right")
        plt.ylabel("Predicted value (μ ± σ)")
        plt.title("Predicted properties")
        path = os.path.join(self.out_dir, image_name)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def build(
        self,
        role: str,
        psmiles: str,
        neighbors: List[Dict],
        predictions: Dict[str, Dict[str, float]],
        candidates: List[Dict],
        evidence: List[Dict],
        extra_notes: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Returns (markdown_report, figure_path)
        """
        figure_path = None
        if predictions:
            img_name = f"props_{psmiles.replace('/', '_')}_{_now_iso().replace(':','-')}.png"
            figure_path = self._plot_properties(predictions, img_name)

        header = f"# Polymer Analysis Report\n\n**Input (P-SMILES):** `{psmiles}`\n**Role:** {role}\n**Generated:** {_now_iso()}\n\n"
        nb = ""
        if neighbors:
            nb += "## Nearest Neighbors (vector similarity)\n"
            for n in neighbors:
                nb += f"- `{n.get('psmiles','?')}` (similarity: {n.get('similarity',0):.3f})\n"
            nb += "\n"

        pp = ""
        if predictions:
            pp += "## Property Predictions (μ ± σ)\n"
            for k, v in predictions.items():
                pp += f"- **{k}**: {_fmt_mu_sigma(v['mu'], v.get('sigma', 0.0))}\n"
            if figure_path:
                pp += f"\n![Predicted properties]({figure_path})\n\n"

        gen = ""
        if candidates:
            gen += "## Generated Candidates\n"
            for c in candidates:
                line = f"- `{c.get('psmiles','?')}`"
                if "score" in c:
                    line += f" (score: {c['score']:.3f})"
                gen += line + "\n"
            gen += "\n"

        ev = ""
        if evidence:
            ev += "## Retrieved Evidence (web)\n"
            for e in evidence[:8]:
                ev += f"- **{e.get('title','(no title)')}** — {e.get('url','')}\n  \n  {e.get('snippet','')[:400]}…\n"
            ev += "\n"

        nx = ""
        if role.lower().startswith("non"):
            nx += "## Plain-language Summary\n"
            nx += (
                "This polymer description was analyzed using a multimodal encoder and property predictors. "
                "We also searched publicly available sources for use-cases, processing guidance, and safety notes. "
                "Values include uncertainty (σ) to reflect model confidence.\n\n"
            )
        else:
            nx += "## Technical Notes\n"
            nx += (
                "Predictions reflect the current CL encoder + downstream heads; uncertainties derived from ensemble/aleatoric settings "
                "if available. Retrieved evidence is not curated and should be considered background; verify against handbooks or primary data.\n\n"
            )

        if extra_notes:
            nx += f"**Additional Notes:** {extra_notes}\n\n"

        md = header + nb + pp + gen + ev + nx
        return md, figure_path
