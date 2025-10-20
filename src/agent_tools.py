"""LangChain tool definitions for the polymer foundation models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from langchain.tools import tool

from Data_Modalities import AdvancedPolymerMultimodalExtractor
from rag_pipeline import build_retriever

from .polymer_services import (
    PolymerSample,
    SimplePSMILESTokenizer,
    combine_modal_embeddings,
    encode_polymers,
    load_contrastive_model,
    parse_polymer_record,
)

_TOKENIZER = SimplePSMILESTokenizer(max_length=128)
_MODEL = load_contrastive_model()
_RAG_RETRIEVER = None


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file {path} does not exist")
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _to_samples(records: Iterable[Dict]) -> List[PolymerSample]:
    return [parse_polymer_record(rec) for rec in records]


def _ensure_retriever():
    global _RAG_RETRIEVER
    if _RAG_RETRIEVER is None:
        guidelines_path = Path("Data/Guidelines")
        _RAG_RETRIEVER = build_retriever(papers_path=str(guidelines_path))
    return _RAG_RETRIEVER


@tool("polymer_modality_processing_tool")
def polymer_modality_processing_tool(csv_path: str, chunk_size: int = 500, num_workers: int = 4) -> str:
    """Process a polymer CSV to populate graph/geometry/fingerprint columns.

    Args:
        csv_path: Path to the raw polymer CSV file.
        chunk_size: Number of rows processed per chunk.
        num_workers: Number of parallel workers.
    Returns:
        Path to the processed CSV ("*_processed.csv").
    """

    extractor = AdvancedPolymerMultimodalExtractor(csv_path)
    extractor.process_all_polymers_parallel(chunk_size=chunk_size, num_workers=num_workers)
    processed = Path(csv_path).with_name(Path(csv_path).stem + "_processed.csv")
    return str(processed)


@tool("contrastive_embedding_tool")
def contrastive_embedding_tool(samples_path: str) -> Dict[str, List[float]]:
    """Encode polymers into the shared CL embedding space.

    ``samples_path`` must point to a JSONL file with fields produced by the
    modality processor (``graph``, ``geometry``, ``fingerprints``, ``psmiles``).
    """

    records = _load_jsonl(Path(samples_path))
    samples = _to_samples(records)
    embeddings = encode_polymers(samples, _TOKENIZER, _MODEL)
    combined = combine_modal_embeddings(embeddings)
    if combined.shape[0] != len(samples):
        raise RuntimeError("Embedding batch mismatch")
    return {samples[i].name: combined[i].tolist() for i in range(len(samples))}


def _train_linear_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    weights, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    return weights


def _predict_linear_regression(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return x_aug @ weights


@tool("polymer_property_prediction_tool")
def polymer_property_prediction_tool(dataset_path: str, target_key: str, train_fraction: float = 0.8) -> Dict[str, float]:
    """Fit a lightweight regression head on top of CL embeddings to predict properties."""

    records = _load_jsonl(Path(dataset_path))
    samples = _to_samples(records)
    targets = []
    for record in records:
        if target_key not in record:
            raise KeyError(f"Target key '{target_key}' missing from record {record}")
        targets.append(float(record[target_key]))
    y = np.array(targets, dtype=np.float32)
    embeddings = encode_polymers(samples, _TOKENIZER, _MODEL)
    x = combine_modal_embeddings(embeddings)
    n_train = max(1, int(len(samples) * train_fraction))
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    weights = _train_linear_regression(x_train, y_train)
    preds = _predict_linear_regression(x_test, weights) if len(x_test) else np.array([])
    metrics = {}
    if len(x_test):
        mse = float(np.mean((preds - y_test) ** 2))
        mae = float(np.mean(np.abs(preds - y_test)))
        metrics.update({"mse": mse, "mae": mae})
    metrics["train_samples"] = int(n_train)
    metrics["test_samples"] = int(len(x) - n_train)
    if len(x_test):
        metrics["predictions"] = preds.tolist()
        metrics["ground_truth"] = y_test.tolist()
    return metrics


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


@tool("polymer_generation_tool")
def polymer_generation_tool(seed_path: str, library_path: str, top_k: int = 5) -> List[Dict[str, float]]:
    """Retrieve similar polymers to a seed polymer using CL embeddings."""

    seed_records = _load_jsonl(Path(seed_path))
    if len(seed_records) != 1:
        raise ValueError("Seed file must contain exactly one polymer record")
    seed_sample = _to_samples(seed_records)[0]

    library_records = _load_jsonl(Path(library_path))
    library_samples = _to_samples(library_records)

    seed_emb = combine_modal_embeddings(encode_polymers([seed_sample], _TOKENIZER, _MODEL))
    library_emb = combine_modal_embeddings(encode_polymers(library_samples, _TOKENIZER, _MODEL))

    sims = _cosine_similarity(seed_emb, library_emb)[0]
    idx = np.argsort(-sims)[:top_k]
    results: List[Dict[str, float]] = []
    for i in idx:
        results.append(
            {
                "name": library_samples[i].name,
                "psmiles": library_samples[i].psmiles,
                "similarity": float(sims[i]),
            }
        )
    return results


@tool("polymer_guideline_search_tool")
def polymer_guideline_search_tool(query: str, top_k: int = 4) -> str:
    """Search the local polymer knowledge base using the RAG ensemble retriever."""

    retriever = _ensure_retriever()
    docs = retriever.invoke(query)
    snippets = [doc.page_content for doc in docs[:top_k]]
    return "\n\n".join(snippets)


all_tools = [
    polymer_modality_processing_tool,
    contrastive_embedding_tool,
    polymer_property_prediction_tool,
    polymer_generation_tool,
    polymer_guideline_search_tool,
]
