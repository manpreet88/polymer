"""Simple polymer generation utilities leveraging the knowledge base and predictors."""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from rdkit import Chem

from .datamodels import GeneratedPolymer, ProcessedPolymer, PropertyPrediction
from .embedding_service import MultimodalEmbeddingService
from .knowledge_base import PolymerKnowledgeBase
from .property_predictor import PropertyPredictorEnsemble
from .data_ingestion import PolymerDataIngestionTool

LOGGER = logging.getLogger(__name__)


def _mutate_psmiles(psmiles: str, *, max_attempts: int = 5) -> Optional[str]:
    """Produce a lightweight mutation of a PSMILES string using RDKit."""

    base = psmiles.replace("[*]", "C")
    mol = Chem.MolFromSmiles(base)
    if mol is None:
        return None
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in (6, 7, 8)]
    if not atoms:
        return None
    for _ in range(max_attempts):
        atom = random.choice(atoms)
        editable = Chem.RWMol(mol)
        idx = atom.GetIdx()
        new_atom = editable.AddAtom(Chem.Atom(6))
        try:
            editable.AddBond(idx, new_atom, Chem.rdchem.BondType.SINGLE)
            mutated = editable.GetMol()
            Chem.SanitizeMol(mutated)
            mutated_smiles = Chem.MolToSmiles(mutated, canonical=True)
            return mutated_smiles
        except Exception:
            continue
    return None


class PolymerGeneratorTool:
    """Generate candidate polymers by retrieving and mutating knowledge base entries."""

    def __init__(
        self,
        embedding_service: MultimodalEmbeddingService,
        knowledge_base: PolymerKnowledgeBase,
        predictor: Optional[PropertyPredictorEnsemble] = None,
        ingestion_tool: Optional[PolymerDataIngestionTool] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.knowledge_base = knowledge_base
        self.predictor = predictor
        self.ingestion_tool = ingestion_tool or PolymerDataIngestionTool()

    def generate(
        self,
        seed_polymer: ProcessedPolymer,
        *,
        top_k: int = 5,
        mutations: int = 3,
    ) -> List[GeneratedPolymer]:
        """Return retrieval-augmented generations for a seed polymer."""

        embedding = self.embedding_service.embed_polymer(seed_polymer)
        results = self.knowledge_base.search(embedding.vector, top_k=top_k)
        generated: List[GeneratedPolymer] = []

        for entry, score in results:
            metadata: Dict = {
                "similarity": score,
                "retrieved_from": entry.source,
                "mode": "retrieval",
            }
            predictions: List[PropertyPrediction] = []
            if self.predictor is not None:
                try:
                    predictions = self.predictor.predict_polymer(seed_polymer)
                except Exception as exc:
                    LOGGER.warning("Property prediction failed during generation: %s", exc)
            generated.append(
                GeneratedPolymer(
                    psmiles=entry.psmiles,
                    metadata=metadata,
                    property_estimates=predictions,
                )
            )

        if mutations > 0:
            for entry, score in results[:mutations]:
                mutated = _mutate_psmiles(entry.psmiles)
                if mutated is None:
                    continue
                metadata = {
                    "parent": entry.psmiles,
                    "mode": "mutation",
                    "similarity": score,
                }
                try:
                    mutated_polymer = self.ingestion_tool.ingest_psmiles(mutated, source="mutation", metadata={"parent": entry.psmiles})
                except Exception as exc:
                    LOGGER.warning("Failed to ingest mutated polymer %s: %s", mutated, exc)
                    continue
                predictions: List[PropertyPrediction] = []
                if self.predictor is not None:
                    try:
                        predictions = self.predictor.predict_polymer(mutated_polymer)
                    except Exception as exc:
                        LOGGER.warning("Property prediction failed for mutated polymer %s: %s", mutated, exc)
                generated.append(GeneratedPolymer(psmiles=mutated, metadata=metadata, property_estimates=predictions))
        return generated


__all__ = ["PolymerGeneratorTool"]
