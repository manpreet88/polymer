# src/rag_pipeline.py
# -*- coding: utf-8 -*-

"""
Polymer RAG pipeline (extensive edition)

Features
--------
- Fetch thousands of OA PDFs from OpenAlex + arXiv (no API keys).
- Parallel downloads with retries/backoff; de-dup via SHA256; manifest.jsonl.
- Rich metadata (title, venue, year, url, source) attached to chunks.
- BM25 + Vector ensemble using local RRF (robust across LangChain versions).
- Embeddings: "sentence-transformers/all-mpnet-base-v2" (default) or "intfloat/e5-large-v2"
  with correct query/passage prefixing handled for you.
- Vector store: Chroma (default) or FAISS (switch via `vector_backend`).

Install
-------
pip install -U \
  requests tqdm \
  langchain>=0.1 langchain-community langchain-text-splitters \
  chromadb>=0.5 sentence-transformers>=2.7 pypdf>=4.2

# Optional if using FAISS:
# pip install faiss-cpu

Usage (big build)
-----------------
from src.rag_pipeline import build_retriever_from_web, POLYMER_KEYWORDS

retriever = build_retriever_from_web(
    polymer_keywords=POLYMER_KEYWORDS,
    max_arxiv=300,
    max_openalex=3000,
    from_year=2010,
    vector_backend="chroma",              # or "faiss"
    embedding_model="intfloat/e5-large-v2",  # or "sentence-transformers/all-mpnet-base-v2"
    persist_dir="chroma_polymer_db_big",
    k=6,
)

docs = retriever.get_relevant_documents("foundation models for polymer property prediction")

Notes
-----
- Be polite with very large crawls. OpenAlex per_page max is 200.
- manifest.jsonl is written in the download directory so you can resume/rebuild easily.
"""

from __future__ import annotations
import os
import re
import time
import json
import hashlib
import pathlib
import tempfile
from typing import List, Optional, Dict, Any, Iterable, Union

import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain (compatible with 0.1.x and 0.2+ splits)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.retrievers import BM25Retriever

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

ARXIV_SEARCH_URL = "http://export.arxiv.org/api/query"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"  # keyless

DEFAULT_PERSIST_DIR = "chroma_polymer_db"
DEFAULT_TMP_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "polymer_rag_pdfs")

# Polymer-centric keywords (broad & subfields, deduped later)
POLYMER_KEYWORDS = [
    # Core
    "polymer", "macromolecule", "macromolecular", "polymeric",
    "polymer informatics", "polymer chemistry", "polymer physics",

    # Representations / notations
    "BigSMILES", "PSMILES", "pSMILES", "polymer SMILES", "polymer sequence",
    "stochastic graph grammar polymer", "polymer graph",

    # ML / foundation models
    "foundation model polymer", "masked language model polymer", "polymer LLM",
    "self-supervised polymer", "transformer polymer", "Perceiver polymer", "Performer polymer",
    "representation learning polymer", "contrastive learning polymer",

    # Electrolytes & energy
    "polymer electrolyte", "solid polymer electrolyte", "ionogel",
    "block copolymer electrolyte", "polymer electrolyte membrane", "ion-conducting polymer",

    # Processing/structure
    "rheology polymer", "morphology polymer", "crystallinity polymer",
    "self-assembly polymer", "phase separation polymer",

    # Mechanics/thermal
    "glass transition polymer", "Tg polymer", "fracture polymer",
    "viscoelastic polymer", "creep polymer",

    # Sustainability
    "recyclable polymer", "depolymerization", "biopolymer",

    # Generative / property prediction
    "generative model polymer", "inverse design polymer",
    "property prediction polymer",
]

# --------------------------------------------------------------------------------------
# Helpers: hashing, filenames, dirs, manifest
# --------------------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _safe_filename(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:180]

def _is_probably_pdf(raw: bytes, content_type: str = "") -> bool:
    if raw[:4] == b"%PDF":
        return True
    return "pdf" in (content_type or "").lower()

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _dedup_seen(item: str, seen: set) -> bool:
    if item in seen:
        return True
    seen.add(item)
    return False

def _append_manifest(out_dir: str, record: Dict[str, Any]):
    try:
        with open(os.path.join(out_dir, "manifest.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _load_manifest(out_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns a mapping from local filepath -> metadata dict (if manifest exists).
    """
    mpath = os.path.join(out_dir, "manifest.jsonl")
    data: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(mpath):
        return data
    with open(mpath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                p = rec.get("path")
                if p:
                    data[p] = rec
            except Exception:
                continue
    return data

# --------------------------------------------------------------------------------------
# Downloading PDFs (single + parallel with retry)
# --------------------------------------------------------------------------------------

def download_pdf(url: str, out_dir: str, suggested_name: Optional[str] = None,
                 timeout: int = 60, meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Download a PDF and return local file path, or None on failure.
    Deduplicates by SHA256 content hash. Writes manifest record if meta provided.
    """
    try:
        headers = {"User-Agent": "polymer-rag/1.0 (+https://example.local)"}
        with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            raw = r.content
        if not raw or not _is_probably_pdf(raw, content_type):
            return None

        sha = _sha256_bytes(raw)
        _ensure_dir(out_dir)

        existing = list(pathlib.Path(out_dir).glob(f"*{sha[:16]}*.pdf"))
        if existing:
            path = str(existing[0])
            if meta:
                rec = dict(meta)
                rec.update({"sha256": sha, "path": path})
                _append_manifest(out_dir, rec)
            return path

        base = suggested_name or pathlib.Path(url).name or "paper.pdf"
        base = _safe_filename(base)
        if not base.lower().endswith(".pdf"):
            base += ".pdf"
        fname = f"{sha[:16]}_{base}"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(raw)

        if meta:
            rec = dict(meta)
            rec.update({"sha256": sha, "path": fpath})
            _append_manifest(out_dir, rec)

        return fpath
    except Exception:
        return None

def _retry(fn, *args, _retries=3, _sleep=0.5, **kwargs):
    for i in range(_retries):
        out = fn(*args, **kwargs)
        if out:
            return out
        time.sleep(_sleep * (2 ** i))
    return None

def _download_one(entry: Union[str, Dict[str, Any]], out_dir: str):
    if isinstance(entry, dict):
        return download_pdf(entry["url"], out_dir, suggested_name=entry.get("name"), meta=entry.get("meta"))
    return download_pdf(entry, out_dir)

def parallel_download_pdfs(entries: List[Union[str, Dict[str, Any]]], out_dir: str, max_workers: int = 16) -> List[str]:
    _ensure_dir(out_dir)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_retry, _download_one, e, out_dir) for e in entries]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Downloading PDFs (parallel)"):
            p = f.result()
            if p:
                results.append(p)
    return results

# --------------------------------------------------------------------------------------
# arXiv (keyless)
# --------------------------------------------------------------------------------------

def _arxiv_query_from_keywords(keywords: List[str]) -> str:
    kw = [k.replace('"', '') for k in keywords]
    # Keep query compact; search title/abstract; include relevant categories
    terms = " OR ".join([f'ti:"{k}"' for k in kw] + [f'abs:"{k}"' for k in kw])
    cats = "(cat:cond-mat.mtrl-sci OR cat:cond-mat.soft OR cat:physics.chem-ph OR cat:cs.LG OR cat:stat.ML)"
    return f"({terms}) AND {cats}"

def fetch_arxiv_pdf_urls(keywords: List[str], max_results: int = 100, sort_by: str = "submittedDate") -> List[str]:
    query = _arxiv_query_from_keywords(keywords)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    headers = {"User-Agent": "polymer-rag/1.0 (+https://example.local)"}
    resp = requests.get(ARXIV_SEARCH_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    xml = resp.text
    urls = re.findall(r'href="(https?://arxiv\.org/pdf/[^"]+)"', xml)
    seen = set()
    return [u for u in urls if not _dedup_seen(u, seen)]

def fetch_arxiv_pdfs(keywords: List[str], out_dir: str, max_results: int = 100, polite_delay: float = 0.25) -> List[str]:
    urls = fetch_arxiv_pdf_urls(keywords, max_results=max_results)
    entries = []
    for url in urls:
        suggested = url.rstrip("/").split("/")[-1]
        entries.append({"url": url, "name": suggested, "meta": {"source": "arxiv", "url": url}})
    # Parallel (still kind to arXiv)
    paths = parallel_download_pdfs(entries, out_dir, max_workers=8)
    time.sleep(polite_delay)  # small idle gap after batch
    return paths

# --------------------------------------------------------------------------------------
# OpenAlex (keyless, big pagination)
# --------------------------------------------------------------------------------------

def _openalex_build_search_query(keywords: List[str]) -> str:
    return " ".join(sorted(set(keywords), key=str.lower))

def _openalex_fetch_works(
    keywords: List[str],
    max_results: int = 5000,
    per_page: int = 200,         # OpenAlex max page size
    from_year: int = 2000,
    allowed_types: List[str] = ("journal-article", "posted-content"),
) -> List[Dict[str, Any]]:
    search = _openalex_build_search_query(keywords)
    page = 1
    works: List[Dict[str, Any]] = []
    headers = {"User-Agent": "polymer-rag/1.0 (+https://example.local)"}

    filters = [
        "is_oa:true",
        "language:en",
        f"from_publication_date:{from_year}-01-01",
        "type:" + "|".join(allowed_types),
    ]
    filter_str = ",".join(filters)

    while len(works) < max_results:
        params = {
            "search": search,
            "per_page": per_page,
            "page": page,
            "sort": "publication_date:desc",
            "filter": filter_str,
        }
        resp = requests.get(OPENALEX_WORKS_URL, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        works.extend(results)
        if len(results) < per_page:
            break
        page += 1
        time.sleep(0.15)
        if len(works) >= max_results:
            works = works[:max_results]
            break

    return works

def _openalex_extract_pdf_entries(works: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns list of download entries:
      {"url": pdf_url, "name": filename_hint, "meta": {title, year, venue, url, source}}
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for w in works:
        best = (w.get("best_oa_location") or {})
        pdf = best.get("url_for_pdf") or ""
        if not pdf:
            pl = w.get("primary_location") or {}
            pdf = pl.get("pdf_url") or ""
        if not pdf or pdf in seen:
            continue
        seen.add(pdf)
        title = (w.get("title") or "paper").strip()
        year = w.get("publication_year")
        venue = ((w.get("host_venue") or {}).get("display_name") or "").strip()
        name = " - ".join([s for s in [title, venue, str(year or "")] if s])
        out.append({
            "url": pdf,
            "name": name,
            "meta": {"title": title, "year": year, "venue": venue, "url": pdf, "source": "openalex"}
        })
    return out

def fetch_openalex_pdfs(
    keywords: List[str],
    out_dir: str,
    max_results: int = 5000,
    per_page: int = 200,
    from_year: int = 2000,
) -> List[str]:
    works = _openalex_fetch_works(keywords, max_results=max_results, per_page=per_page, from_year=from_year)
    entries = _openalex_extract_pdf_entries(works)
    return parallel_download_pdfs(entries, out_dir, max_workers=16)

# --------------------------------------------------------------------------------------
# Embeddings: E5 query/passage prefixing (optional)
# --------------------------------------------------------------------------------------

class SmartHFEmbeddings(HuggingFaceEmbeddings):
    """
    Drop-in HuggingFaceEmbeddings with optional E5 prefix handling:
      - If model_name contains 'e5', we prefix queries with 'query: ' and documents with 'passage: '.
      - Otherwise behaves like base class.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._use_e5 = "e5" in (model_name or "").lower()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._use_e5:
            texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self._use_e5:
            text = f"query: {text}"
        return super().embed_query(text)

# --------------------------------------------------------------------------------------
# Local Ensemble (RRF) — no langchain.retrievers import needed
# --------------------------------------------------------------------------------------

class SimpleEnsembleRetriever:
    """
    Minimal, LC-compatible retriever that merges results from multiple retrievers
    using Reciprocal Rank Fusion (RRF), with optional weights.
    """

    def __init__(self, retrievers, weights=None, k: int = 5, rrf_k: int = 60):
        assert retrievers, "At least one base retriever required"
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        assert len(self.weights) == len(self.retrievers)
        self.k = k
        self.rrf_k = rrf_k  # standard RRF constant

    def _run_retriever(self, retriever, query: str):
        # Try common APIs across LangChain versions, in a robust order.
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)  # Runnable-style
        if callable(retriever):
            return retriever(query)
        if hasattr(retriever, "_get_relevant_documents"):
            # Older LC BM25 can require run_manager kw-only; handle both.
            try:
                return retriever._get_relevant_documents(query, run_manager=None)
            except TypeError:
                try:
                    return retriever._get_relevant_documents(query)
                except TypeError:
                    pass
        raise TypeError(f"Unsupported retriever interface: {type(retriever)}")

    def get_relevant_documents(self, query: str):
        # 1) collect results from each retriever
        all_lists = []
        for r in self.retrievers:
            docs = self._run_retriever(r, query)
            all_lists.append(docs or [])

        # 2) RRF scoring
        scores: Dict[int, float] = {}
        index_map: Dict[int, Any] = {}

        def doc_key(doc):
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source", "")
            page = str(meta.get("page", ""))
            text = (getattr(doc, "page_content", "") or "")[:500]
            return f"{src}|{page}|{hash(text)}"

        key_to_idx: Dict[str, int] = {}
        next_idx = 0

        for w, docs in zip(self.weights, all_lists):
            for rank, doc in enumerate(docs):
                key = doc_key(doc)
                if key not in key_to_idx:
                    key_to_idx[key] = next_idx
                    index_map[next_idx] = doc
                    next_idx += 1
                idx = key_to_idx[key]
                scores[idx] = scores.get(idx, 0.0) + w * (1.0 / (self.rrf_k + rank + 1))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [index_map[i] for i, _ in ranked[: self.k]]

# --------------------------------------------------------------------------------------
# KB builder (local or from web)
# --------------------------------------------------------------------------------------

def _attach_extra_metadata_from_manifest(docs: List[Any], manifest: Dict[str, Dict[str, Any]]) -> None:
    """
    Enrich each Document's metadata using the download manifest (title, year, venue, url, source).
    """
    for d in docs:
        src_path = d.metadata.get("source", "")
        if not src_path:
            continue
        rec = manifest.get(src_path)
        if not rec:
            # Try to match by stem (sometimes loaders normalize path)
            for k, v in manifest.items():
                if os.path.basename(k) == os.path.basename(src_path):
                    rec = v
                    break
        if rec:
            # Propagate useful metadata
            for k in ("title", "year", "venue", "url", "source"):
                if k in rec:
                    d.metadata[k] = rec[k]

def _split_and_build_retriever(
    documents_dir: str,
    persist_dir: Optional[str] = None,
    k: int = 6,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    vector_backend: str = "chroma",   # "chroma" or "faiss"
    min_chunk_chars: int = 200,
):
    print(f"🗂️  Loading PDFs from: {documents_dir}")
    loader = DirectoryLoader(
        documents_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError("No PDF documents found to index.")

    # Attach extra metadata (title/year/venue/url/source) from manifest
    manifest = _load_manifest(documents_dir)
    _attach_extra_metadata_from_manifest(docs, manifest)

    # Chunking tuned for scientific PDFs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=250,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    documents = text_splitter.split_documents(docs)
    # Drop tiny chunks
    documents = [d for d in documents if len(d.page_content or "") >= min_chunk_chars]

    # BM25 (keyword) — give it slightly higher k than final fusion k
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = max(k, 8)

    # Vector store + embeddings
    print(f"🔤 Using embeddings: {embedding_model}")
    embeddings = SmartHFEmbeddings(model_name=embedding_model)

    if vector_backend.lower() == "chroma":
        if persist_dir:
            print(f"💾 Using Chroma (persist_dir={persist_dir})")
            vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)
            # Chroma >= 0.4 auto-persists; guard for older versions
            try:
                vector_store.persist()
            except Exception:
                pass
        else:
            vector_store = Chroma.from_documents(documents, embeddings)
    elif vector_backend.lower() == "faiss":
        try:
            from langchain_community.vectorstores import FAISS
        except Exception as e:
            raise RuntimeError("FAISS backend requested but faiss is not installed. Run `pip install faiss-cpu`.") from e
        print("💾 Using FAISS (in-memory; serialize with FAISS.save_local if desired)")
        vector_store = FAISS.from_documents(documents, embeddings)
    else:
        raise ValueError("vector_backend must be 'chroma' or 'faiss'")

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Local ensemble (RRF)
    ensemble_retriever = SimpleEnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.45, 0.55],  # slight bias to semantic retrieval
        k=k,
    )
    print("✅ RAG knowledge base ready (Ensemble via local RRF).")
    return ensemble_retriever

def build_retriever_from_web(
    polymer_keywords: Optional[List[str]] = None,
    max_arxiv: int = 300,
    max_openalex: int = 3000,
    from_year: int = 2010,
    extra_pdf_urls: Optional[List[str]] = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    tmp_download_dir: str = DEFAULT_TMP_DOWNLOAD_DIR,
    k: int = 6,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    vector_backend: str = "chroma",
    max_workers: int = 16,
):
    """
    Build an extensive ensemble retriever by fetching polymer PDFs from:
      - OpenAlex OA (massive, keyless) with year floor
      - arXiv (keyless)
      - optional direct URLs
    """
    # Dedup and normalize keywords
    polymer_keywords = sorted(set(polymer_keywords or POLYMER_KEYWORDS), key=str.lower)

    print("📡 Fetching polymer PDFs from the web (OpenAlex + arXiv)...")
    _ensure_dir(tmp_download_dir)

    # OpenAlex first (largest yield)
    openalex_paths = fetch_openalex_pdfs(
        polymer_keywords, out_dir=tmp_download_dir,
        max_results=max_openalex, per_page=200, from_year=from_year
    )

    # arXiv
    arxiv_paths = fetch_arxiv_pdfs(
        polymer_keywords, out_dir=tmp_download_dir, max_results=max_arxiv
    )

    # Extra direct URLs (if any)
    extra_paths: List[str] = []
    if extra_pdf_urls:
        extra_entries = [{"url": u, "name": None, "meta": {"url": u, "source": "extra"}} for u in extra_pdf_urls]
        extra_paths = parallel_download_pdfs(extra_entries, tmp_download_dir, max_workers=max_workers)

    total = len(openalex_paths) + len(arxiv_paths) + len(extra_paths)
    print(f"✅ Downloaded {total} PDFs "
          f"({len(arxiv_paths)} arXiv, {len(openalex_paths)} OpenAlex OA, {len(extra_paths)} extra).")
    if total == 0:
        raise RuntimeError("No PDFs fetched. Adjust keywords or add extra_pdf_urls.")

    print("🧠 Building the knowledge base (BM25 + Vector)...")
    retriever = _split_and_build_retriever(
        documents_dir=tmp_download_dir,
        persist_dir=persist_dir,
        k=k,
        embedding_model=embedding_model,
        vector_backend=vector_backend,
    )
    return retriever

def build_retriever(
    papers_path: str,
    persist_dir: Optional[str] = DEFAULT_PERSIST_DIR,
    k: int = 6,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    vector_backend: str = "chroma",
):
    """Build from local PDFs (kept for compatibility)."""
    print("📚 Building RAG knowledge base from local PDFs...")
    return _split_and_build_retriever(
        documents_dir=papers_path,
        persist_dir=persist_dir,
        k=k,
        embedding_model=embedding_model,
        vector_backend=vector_backend,
    )

def build_retriever_polymer_foundation_models(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    k: int = 6,
    from_year: int = 2015,
    vector_backend: str = "chroma",
):
    """Convenience wrapper focusing on foundation models & polymer representations."""
    fm_kw = list(set(POLYMER_KEYWORDS + [
        "BigSMILES", "PSMILES", "polymer SMILES", "polymer language model",
        "foundation model polymer", "masked language model polymer",
        "self-supervised polymer", "generative polymer",
        "Perceiver polymer", "Performer polymer",
        "polymer sequence modeling", "representation learning polymer",
    ]))
    return build_retriever_from_web(
        polymer_keywords=fm_kw,
        max_arxiv=400,
        max_openalex=4000,
        from_year=from_year,
        persist_dir=persist_dir,
        k=k,
        vector_backend=vector_backend,
    )

# --------------------------------------------------------------------------------------
# CLI smoke (disabled by default, enable if you want a quick check)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = build_retriever_from_web(
        polymer_keywords=POLYMER_KEYWORDS,
        max_arxiv=150,
        max_openalex=1500,
        from_year=2010,
        persist_dir="chroma_polymer_db_big",
        k=6,
        embedding_model="intfloat/e5-large-v2",   # or "sentence-transformers/all-mpnet-base-v2"
        vector_backend="chroma",                  # or "faiss"
    )
    print("🔎 Sample query:")
    results = retriever.get_relevant_documents("PSMILES for polymer electrolyte design")
    for i, d in enumerate(results, 1):
        meta = d.metadata
        src = meta.get("source", "unknown")
        title = meta.get("title") or os.path.basename(meta.get("source", "")) or "document"
        year = meta.get("year", "")
        print(f"[{i}] {title} ({year}) [{src}] :: {(d.page_content or '')[:180]}...")
