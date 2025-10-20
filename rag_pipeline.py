# src/rag_pipeline.py
# -*- coding: utf-8 -*-

"""
Polymer RAG pipeline (extensive edition, hardened + more OA sources)

What’s included
---------------
- Keyless web harvesting from:
  • OpenAlex (OA-only, multiple types, robust PDF URL extraction)
  • arXiv (Atom XML parsing for <link type="application/pdf">)
  • Europe PMC (OA articles + PMC full text; extracts direct PDF URLs)
  • Optional: direct PDF URLs (extra_pdf_urls)
  • Optional: cautious DOI resolution (allowlisted OA domains only)

- Parallel downloads with retries/backoff; SHA256 de-dup; manifest.jsonl
- Rich metadata (title, venue, year, url, origin) attached to chunks
- Ensemble retrieval: BM25 + Vector (Chroma or FAISS) fused with RRF
- Embeddings: "sentence-transformers/all-mpnet-base-v2" (default)
             or "intfloat/e5-large-v2" (auto query/passsage prefixing)
- LangChain 0.1+ compatible (retriever invocation paths covered)

Install
-------
pip install -U \
  requests tqdm \
  langchain>=0.1 langchain-community langchain-text-splitters \
  chromadb>=0.5 sentence-transformers>=2.7 pypdf>=4.2

# Optional if using FAISS:
# pip install faiss-cpu

Quick use (big web build)
-------------------------
from src.rag_pipeline import build_retriever_from_web, POLYMER_KEYWORDS

retriever = build_retriever_from_web(
    polymer_keywords=POLYMER_KEYWORDS,
    max_arxiv=300,
    max_openalex=3000,
    max_europepmc=3000,
    from_year=2010,
    vector_backend="chroma",                  # or "faiss"
    embedding_model="intfloat/e5-large-v2",   # or "sentence-transformers/all-mpnet-base-v2"
    persist_dir="chroma_polymer_db_big",
    k=6,
)

docs = retriever.get_relevant_documents("foundation models for polymer property prediction")

Notes
-----
- Be polite with very large crawls (tiny sleeps are built-in).
- manifest.jsonl is written in the download directory for resume/debug.
- For OA publisher PDFs like Nature Communications, Europe PMC results
  often include a direct PDF URL when it’s truly OA.
"""

from __future__ import annotations
import os
import re
import time
import json
import hashlib
import pathlib
import tempfile
import logging
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any, Iterable, Union, Tuple

import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.retrievers import BM25Retriever

# --------------------------------------------------------------------------------------
# Logging / Config
# --------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("polymer-rag")

# Endpoints
ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"        # Atom XML
OPENALEX_WORKS_URL = "https://api.openalex.org/works"          # JSON
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULT_PERSIST_DIR = "chroma_polymer_db"
DEFAULT_TMP_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "polymer_rag_pdfs")

# OA allowlist for cautious DOI resolution (optional)
PDF_DOMAIN_ALLOWLIST = {
    "nature.com",            # includes Nature Communications OA PDFs
    "springernature.com",
    "springer.com",          # some OA journals
    "frontiersin.org",
    "mdpi.com",
    "plos.org",
    "royalsocietypublishing.org",
    "elifesciences.org",
    "biorxiv.org",           # not polymers usually, but harmless
    "chemrxiv.org",
    "arxiv.org",
    "acscatalysis.org",      # sometimes OA
    "pubs.acs.org",          # many paywalled; OA subset works (check headers)
    "rsc.org",               # OA subset
}

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
    "polymer language model", "polymer sequence modeling",

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
    "generative model polymer", "inverse design polymer", "property prediction polymer",
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

def _is_probably_pdf(raw: bytes, content_type: str = "", url_hint: str = "") -> bool:
    if raw[:4] == b"%PDF":
        return True
    if "pdf" in (content_type or "").lower():
        return True
    if url_hint.lower().endswith(".pdf"):
        return True
    return False

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
# HTTP helpers
# --------------------------------------------------------------------------------------

_UA = "polymer-rag/1.2 (+https://example.local)"

def _http_get(url: str, timeout: int = 60, stream: bool = True, accept: Optional[str] = None) -> requests.Response:
    headers = {"User-Agent": _UA}
    if accept:
        headers["Accept"] = accept
    r = requests.get(url, headers=headers, timeout=timeout, stream=stream, allow_redirects=True)
    r.raise_for_status()
    return r

def _download_pdf(url: str, out_dir: str, suggested_name: Optional[str] = None,
                  timeout: int = 60, meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Download a PDF and return local file path, or None on failure.
    Deduplicates by SHA256 content hash. Writes manifest record if meta provided.
    """
    try:
        r = _http_get(url, timeout=timeout, stream=True)
        content_type = r.headers.get("Content-Type", "")
        raw = r.content  # okay since many PDFs are not gigantic; chunking could be added if needed
        if not raw or not _is_probably_pdf(raw, content_type, url):
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
    except Exception as e:
        logger.debug(f"download_pdf failed for {url}: {e}")
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
        return _download_pdf(entry["url"], out_dir, suggested_name=entry.get("name"), meta=entry.get("meta"))
    return _download_pdf(entry, out_dir)

def parallel_download_pdfs(entries: List[Union[str, Dict[str, Any]]], out_dir: str, max_workers: int = 16) -> List[str]:
    _ensure_dir(out_dir)
    if not entries:
        return []
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_retry, _download_one, e, out_dir) for e in entries]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Downloading PDFs (parallel)"):
            p = f.result()
            if p:
                results.append(p)
    return results

# --------------------------------------------------------------------------------------
# arXiv (keyless; robust Atom parsing)
# --------------------------------------------------------------------------------------

def _arxiv_query_from_keywords(keywords: List[str]) -> str:
    kw = [k.replace('"', '') for k in keywords]
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
    resp = _http_get(ARXIV_SEARCH_URL, timeout=60, stream=False, accept="application/atom+xml")
    # requests can't combine params when we pre-composed; re-request with params:
    resp = requests.get(ARXIV_SEARCH_URL, params=params, headers={"User-Agent": _UA, "Accept": "application/atom+xml"}, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    urls: List[str] = []
    for entry in root.findall("a:entry", ns):
        pdf_url = None
        for link in entry.findall("a:link", ns):
            if (link.get("type") == "application/pdf") and link.get("href"):
                pdf_url = link.get("href")
                break
        if pdf_url:
            urls.append(pdf_url)
    # de-dup
    seen = set()
    return [u for u in urls if not _dedup_seen(u, seen)]

def fetch_arxiv_pdfs(keywords: List[str], out_dir: str, max_results: int = 100, polite_delay: float = 0.25) -> List[str]:
    urls = fetch_arxiv_pdf_urls(keywords, max_results=max_results)
    entries = []
    for url in urls:
        suggested = url.rstrip("/").split("/")[-1]
        entries.append({"url": url, "name": suggested, "meta": {"origin": "arxiv", "url": url}})
    paths = parallel_download_pdfs(entries, out_dir, max_workers=8)
    time.sleep(polite_delay)
    return paths

# --------------------------------------------------------------------------------------
# OpenAlex (keyless, broad types, robust PDF extraction)
# --------------------------------------------------------------------------------------

def _openalex_build_search_query(keywords: List[str]) -> str:
    return " ".join(sorted(set(keywords), key=str.lower))

def _openalex_fetch_works(
    keywords: List[str],
    max_results: int = 5000,
    per_page: int = 200,
    from_year: int = 2000,
    allowed_types: List[str] = (
        "journal-article",
        "posted-content",       # preprints
        "proceedings-article",  # conferences
        "book-chapter",
    ),
) -> List[Dict[str, Any]]:
    search = _openalex_build_search_query(keywords)
    page = 1
    works: List[Dict[str, Any]] = []
    headers = {"User-Agent": _UA}

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
        r = requests.get(OPENALEX_WORKS_URL, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
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

def _first_pdf_like(*candidates: Optional[str]) -> Optional[str]:
    for c in candidates:
        if c and c.lower().endswith(".pdf"):
            return c
    for c in candidates:
        if c and ("pdf" in c.lower() or "arxiv.org" in c.lower()):
            return c
    return None

def _openalex_extract_pdf_entries(works: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for w in works:
        pl = (w.get("primary_location") or {})
        best = (w.get("best_oa_location") or {})
        oa = (w.get("open_access") or {})

        pdf = _first_pdf_like(
            best.get("url_for_pdf"),
            pl.get("pdf_url"),
            oa.get("oa_url"),
            (pl.get("source") or {}).get("url")
        )
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
            "meta": {
                "title": title,
                "year": year,
                "venue": venue,
                "url": pdf,
                "origin": "openalex",
            }
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
# Europe PMC (keyless; OA & PMC full text; often publisher PDFs like Nature OA)
# --------------------------------------------------------------------------------------

def _eupmc_build_query(keywords: List[str], from_year: int) -> str:
    """
    Europe PMC advanced query:
      - OA only
      - From specified year
      - Keyword search in title/abstract/fulltext by default
    """
    # Quote each term without using an f-string (avoids backslash-in-expression issue)
    kw = ['"{}"'.format(k.replace('"', '')) for k in keywords]
    kw_expr = " OR ".join(kw) if kw else '"polymer"'
    return f"({kw_expr}) AND OPEN_ACCESS:y AND PUB_YEAR:[{from_year} TO *]"

def _eupmc_extract_pdf_from_result(r: Dict[str, Any]) -> Optional[str]:
    # Priorities:
    # 1) pdfUrl (PMC-hosted OA full text)
    # 2) fullTextUrlList.fullTextUrl where documentStyle == "pdf" or availability == "Open access"
    # See: https://europepmc.org/RestfulWebService
    pdf = r.get("pdfUrl")
    if pdf and pdf.lower().endswith(".pdf"):
        return pdf
    ftlist = (r.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    # Try styles and OA indicator
    for item in ftlist:
        u = item.get("url")
        if not u:
            continue
        style = (item.get("documentStyle") or "").lower()
        avail = (item.get("availability") or "").lower()
        if u.lower().endswith(".pdf") or style == "pdf" or "open" in avail:
            return u
    return None

def _eupmc_fetch_results(keywords: List[str], max_results: int, from_year: int) -> List[Dict[str, Any]]:
    query = _eupmc_build_query(keywords, from_year)
    page_size = 100
    page = 1
    out: List[Dict[str, Any]] = []

    while len(out) < max_results:
        params = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "page": page,
            "sort": "date_desc"
        }
        r = requests.get(EUROPE_PMC_SEARCH_URL, params=params, headers={"User-Agent": _UA}, timeout=60)
        r.raise_for_status()
        data = r.json()
        results = ((data.get("resultList") or {}).get("result") or [])
        if not results:
            break
        out.extend(results)
        if len(results) < page_size:
            break
        page += 1
        time.sleep(0.12)
        if len(out) >= max_results:
            out = out[:max_results]
            break
    return out

def fetch_europepmc_pdfs(
    keywords: List[str],
    out_dir: str,
    max_results: int = 2000,
    from_year: int = 2000,
) -> List[str]:
    results = _eupmc_fetch_results(keywords, max_results=max_results, from_year=from_year)
    entries: List[Dict[str, Any]] = []
    seen = set()
    for r in results:
        pdf = _eupmc_extract_pdf_from_result(r)
        if not pdf or pdf in seen:
            continue
        seen.add(pdf)
        title = (r.get("title") or "paper").strip()
        year = r.get("pubYear") or r.get("firstPublicationDate", "")[:4]
        venue = (r.get("journalTitle") or r.get("bookOrReportDetails") or "").strip()
        name = " - ".join([s for s in [title, venue, str(year or "")] if s])
        entries.append({
            "url": pdf,
            "name": name,
            "meta": {
                "title": title,
                "year": year,
                "venue": venue,
                "url": pdf,
                "origin": "europe_pmc",
            }
        })
    return parallel_download_pdfs(entries, out_dir, max_workers=16)

# --------------------------------------------------------------------------------------
# Optional: DOI resolver (allowlisted domains only, HEAD/GET to confirm PDF)
# --------------------------------------------------------------------------------------

def _domain_allowed(u: str) -> bool:
    try:
        from urllib.parse import urlparse
        netloc = urlparse(u).netloc.lower()
        return any(netloc.endswith(dom) for dom in PDF_DOMAIN_ALLOWLIST)
    except Exception:
        return False

def try_resolve_doi_to_pdf(doi: str, timeout: int = 30) -> Optional[str]:
    """
    Try to resolve a DOI to a direct PDF on an allowlisted OA-friendly domain.
    We first hit doi.org, then follow redirects; if content-type is PDF or URL endswith .pdf, accept.
    """
    try:
        from urllib.parse import quote
        url = f"https://doi.org/{quote(doi)}"
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        final_url = r.url
        if not _domain_allowed(final_url):
            return None
        # Check content-type via HEAD (safer)
        h = requests.head(final_url, headers={"User-Agent": _UA}, timeout=timeout, allow_redirects=True)
        ctype = (h.headers.get("Content-Type") or "").lower()
        if "pdf" in ctype or final_url.lower().endswith(".pdf"):
            return final_url
    except Exception:
        return None
    return None

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
# Local Ensemble (RRF)
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
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)  # Runnable-style
        if callable(retriever):
            return retriever(query)
        if hasattr(retriever, "_get_relevant_documents"):
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
            # DO NOT clobber 'source' (file path). Use it for dedup key.
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
    Enrich each Document's metadata using the download manifest (title, year, venue, url, origin).
    We DO NOT overwrite metadata["source"] which holds the file path from loaders.
    """
    for d in docs:
        src_path = d.metadata.get("source", "")
        if not src_path:
            continue
        rec = manifest.get(src_path)
        if not rec:
            # Try to match by basename (loader may normalize)
            for k, v in manifest.items():
                if os.path.basename(k) == os.path.basename(src_path):
                    rec = v
                    break
        if rec:
            for k in ("title", "year", "venue", "url", "origin"):
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
    documents = [d for d in documents if len(d.page_content or "") >= min_chunk_chars]

    # BM25 (keyword) — slightly higher k than final fusion k
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = max(k, 8)

    # Vector store + embeddings
    print(f"🔤 Using embeddings: {embedding_model}")
    embeddings = SmartHFEmbeddings(model_name=embedding_model)

    if vector_backend.lower() == "chroma":
        if persist_dir:
            print(f"💾 Using Chroma (persist_dir={persist_dir})")
            vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)
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

    # Ensemble (RRF)
    ensemble_retriever = SimpleEnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.45, 0.55],  # slight bias to semantic retrieval
        k=k,
    )
    print("✅ RAG knowledge base ready (Ensemble via local RRF).")
    return ensemble_retriever

def _summarize_counts(tag: str, arxiv: int, openalex: int, eupmc: int, extra: int) -> str:
    return (f"{tag} {arxiv + openalex + eupmc + extra} PDFs "
            f"({arxiv} arXiv, {openalex} OpenAlex OA, {eupmc} Europe PMC, {extra} extra).")

def build_retriever_from_web(
    polymer_keywords: Optional[List[str]] = None,
    max_arxiv: int = 300,
    max_openalex: int = 3000,
    max_europepmc: int = 3000,
    from_year: int = 2010,
    extra_pdf_urls: Optional[List[str]] = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    tmp_download_dir: str = DEFAULT_TMP_DOWNLOAD_DIR,
    k: int = 6,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    vector_backend: str = "chroma",
    max_workers: int = 16,
    try_doi_resolution: bool = False,
):
    """
    Build an extensive ensemble retriever by fetching polymer PDFs from:
      - OpenAlex OA (massive, keyless) with year floor
      - arXiv (keyless)
      - Europe PMC (OA publisher/PMC PDFs)
      - optional direct URLs
      - optional cautious DOI resolution (allowlisted OA domains)
    """
    polymer_keywords = sorted(set(polymer_keywords or POLYMER_KEYWORDS), key=str.lower)

    print("📡 Fetching polymer PDFs from the web (OpenAlex + arXiv + Europe PMC)...")
    _ensure_dir(tmp_download_dir)

    # 1) OpenAlex (largest yield)
    openalex_paths = fetch_openalex_pdfs(
        polymer_keywords, out_dir=tmp_download_dir,
        max_results=max_openalex, per_page=200, from_year=from_year
    )

    # 2) arXiv (keyless)
    arxiv_paths = fetch_arxiv_pdfs(
        polymer_keywords, out_dir=tmp_download_dir, max_results=max_arxiv
    )

    # 3) Europe PMC (OA & PMC full text; includes many publisher OA PDFs, e.g., Nature Comms OA)
    eupmc_paths = fetch_europepmc_pdfs(
        polymer_keywords, out_dir=tmp_download_dir, max_results=max_europepmc, from_year=from_year
    )

    # 4) Extra direct URLs (if any)
    extra_paths: List[str] = []
    if extra_pdf_urls:
        extra_entries = [{"url": u, "name": None, "meta": {"url": u, "origin": "extra"}} for u in extra_pdf_urls]
        extra_paths = parallel_download_pdfs(extra_entries, tmp_download_dir, max_workers=max_workers)

    print("✅ " + _summarize_counts("Downloaded", len(arxiv_paths), len(openalex_paths), len(eupmc_paths), len(extra_paths)))
    total = len(openalex_paths) + len(arxiv_paths) + len(eupmc_paths) + len(extra_paths)

    # Optional DOI resolution on small keyword set if still low yield
    if total == 0 and try_doi_resolution:
        print("🔎 Trying cautious DOI resolution on allowlisted OA domains...")
        # heuristics: pick a small Europe PMC run for DOIs
        probe_results = _eupmc_fetch_results(["polymer"], max_results=50, from_year=max(2005, from_year - 5))
        doi_entries = []
        for r in probe_results:
            doi = r.get("doi")
            if not doi:
                continue
            pdf = try_resolve_doi_to_pdf(doi)
            if pdf:
                title = (r.get("title") or "paper").strip()
                year = r.get("pubYear") or r.get("firstPublicationDate", "")[:4]
                venue = (r.get("journalTitle") or "").strip()
                name = " - ".join([s for s in [title, venue, str(year or "")] if s])
                doi_entries.append({
                    "url": pdf,
                    "name": name,
                    "meta": {"title": title, "year": year, "venue": venue, "url": pdf, "origin": "doi"}
                })
        doi_paths = parallel_download_pdfs(doi_entries, tmp_download_dir, max_workers=8)
        total += len(doi_paths)
        print(f"➕ DOI resolution fetched {len(doi_paths)} PDFs (allowlisted OA domains).")

    if total == 0:
        # Fallback pass: broaden queries automatically
        print("⚠️  No PDFs on first pass. Broadening queries and relaxing filters...")
        broad_kw = ["polymer"]
        openalex_paths = fetch_openalex_pdfs(
            broad_kw, out_dir=tmp_download_dir,
            max_results=min(1000, max_openalex), per_page=200, from_year=max(2005, from_year - 5)
        )
        arxiv_paths = fetch_arxiv_pdfs(broad_kw, out_dir=tmp_download_dir, max_results=min(150, max_arxiv))
        eupmc_paths = fetch_europepmc_pdfs(broad_kw, out_dir=tmp_download_dir, max_results=min(1000, max_europepmc), from_year=max(2005, from_year - 5))

        print("🔁 " + _summarize_counts("Fallback fetched", len(arxiv_paths), len(openalex_paths), len(eupmc_paths), 0))
        total = len(openalex_paths) + len(arxiv_paths) + len(eupmc_paths)

        if total == 0:
            raise RuntimeError(
                "No PDFs fetched after fallback. Possible network egress block or API outage.\n"
                "Quick checks:\n"
                "  - Can you curl https://export.arxiv.org/api/query and https://api.openalex.org/works from this env?\n"
                "  - Can you reach https://www.ebi.ac.uk/europepmc/webservices/rest/search ?\n"
                "  - If running in a corporate cluster, allowlist those domains or set the HTTPS proxy.\n"
                "  - You can also supply direct URLs via extra_pdf_urls=[...]."
            )

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
    """Build from local PDFs."""
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
        max_europepmc=4000,
        from_year=from_year,
        persist_dir=persist_dir,
        k=k,
        vector_backend=vector_backend,
    )

# --------------------------------------------------------------------------------------
# CLI smoke (optional)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = build_retriever_from_web(
        polymer_keywords=POLYMER_KEYWORDS,
        max_arxiv=150,
        max_openalex=1500,
        max_europepmc=2000,
        from_year=2010,
        persist_dir="chroma_polymer_db_big",
        k=6,
        embedding_model="intfloat/e5-large-v2",   # or "sentence-transformers/all-mpnet-base-v2"
        vector_backend="chroma",                  # or "faiss"
        try_doi_resolution=False,                 # can set True if still zero-yield in locked envs
    )
    print("🔎 Sample query:")
    results = retriever.get_relevant_documents("PSMILES for polymer electrolyte design")
    for i, d in enumerate(results, 1):
        meta = d.metadata
        src_file = meta.get("source", "unknown-file")
        title = meta.get("title") or os.path.basename(meta.get("source", "")) or "document"
        year = meta.get("year", "")
        origin = meta.get("origin", "")
        print(f"[{i}] {title} ({year}) [{origin}] :: {os.path.basename(src_file)} :: {(d.page_content or '')[:180]}...")
