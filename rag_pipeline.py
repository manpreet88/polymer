# src/rag_pipeline.py
# -*- coding: utf-8 -*-
"""
Polymer RAG pipeline (robust edition)

Features:
- Fetch OA PDFs from OpenAlex + arXiv + Europe PMC (no API keys required).
- Parallel downloads with retries/backoff; de-dup via SHA256; manifest.jsonl to resume.
- Rich metadata attached to saved PDFs.
- BM25 + Vector ensemble via local RRF fusion.
- Embeddings: "sentence-transformers/all-mpnet-base-v2" (default) or "intfloat/e5-large-v2"
  with correct query/passage prefixing handled for you.
- Vector store: Chroma (default) or FAISS (optional).
"""
from __future__ import annotations
import os
import re
import time
import json
import hashlib
import pathlib
import tempfile
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# LangChain / community (expect these installed)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.retrievers import BM25Retriever

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

ARXIV_SEARCH_URL = "http://export.arxiv.org/api/query"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULT_PERSIST_DIR = "chroma_polymer_db"
DEFAULT_TMP_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "polymer_rag_pdfs")
MANIFEST_NAME = "manifest.jsonl"

# default set of polymer-related keywords (expandable)
POLYMER_KEYWORDS = [
    "polymer", "macromolecule", "macromolecular", "polymeric",
    "polymer informatics", "polymer chemistry", "polymer physics",
    "PSMILES", "pSMILES", "BigSMILES", "polymer SMILES", "polymer sequence",
    "foundation model", "self-supervised", "masked language model", "transformer",
    "polymer electrolyte", "polymer morphology", "generative model polymer",
]

# polite defaults
DEFAULT_MAILTO = "your_email@example.com"  # replace if you like

# --------------------------------------------------------------------------------------
# Utility helpers (filenames, hashing, manifest)
# --------------------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(name: str) -> str:
    name = str(name or "").strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", name)
    return name[:200]


def _is_probably_pdf(raw: bytes, content_type: str = "") -> bool:
    if not raw:
        return False
    if raw[:4] == b"%PDF":
        return True
    return "pdf" in (content_type or "").lower()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_manifest(out_dir: str, record: Dict[str, Any]) -> None:
    try:
        _ensure_dir(out_dir)
        with open(os.path.join(out_dir, MANIFEST_NAME), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_manifest(out_dir: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    try:
        mpath = os.path.join(out_dir, MANIFEST_NAME)
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
    except Exception:
        pass
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
        headers = {"User-Agent": f"polymer-rag/1.0 (+{DEFAULT_MAILTO})"}
        with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            raw = r.content
        if not raw or not _is_probably_pdf(raw, content_type):
            return None

        sha = _sha256_bytes(raw)
        _ensure_dir(out_dir)

        # dedup by saved files having hash prefix
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


def _retry(fn, *args, _retries=3, _sleep=0.6, **kwargs):
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


def parallel_download_pdfs(entries: List[Union[str, Dict[str, Any]]], out_dir: str, max_workers: int = 12) -> List[str]:
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
# arXiv helper (robust)
# --------------------------------------------------------------------------------------


def _arxiv_query_from_keywords(keywords: List[str]) -> str:
    kw = [k.replace('"', '') for k in keywords]
    terms = " OR ".join([f'ti:"{k}"' for k in kw] + [f'abs:"{k}"' for k in kw])
    cats = "(cat:cond-mat.mtrl-sci OR cat:cond-mat.soft OR cat:physics.chem-ph OR cat:cs.LG OR cat:stat.ML)"
    return f"({terms}) AND {cats}"


def fetch_arxiv_pdf_urls(keywords: List[str], max_results: int = 200) -> List[str]:
    """
    Extract explicit /pdf/ links and fallback to building from <id> entries.
    """
    query = _arxiv_query_from_keywords(keywords)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": f"polymer-rag/1.0 (+{DEFAULT_MAILTO})"}
    try:
        resp = requests.get(ARXIV_SEARCH_URL, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        xml = resp.text
    except Exception:
        return []

    pdfs = []
    seen = set()
    # explicit /pdf/ hrefs
    for p in re.findall(r'href="(https?://arxiv\.org/pdf/[^"]+)"', xml):
        if p not in seen:
            pdfs.append(p); seen.add(p)
    # fallback: build from <id> entries
    for aid in re.findall(r'<id>(https?://arxiv\.org/abs/[^<]+)</id>', xml):
        m = re.search(r'arxiv\.org\/abs\/([^/]+)(?:/v\d+)?$', aid)
        if m:
            identifier = m.group(1)
            pdf = f"https://arxiv.org/pdf/{identifier}.pdf"
            if pdf not in seen:
                pdfs.append(pdf); seen.add(pdf)
    return pdfs


def fetch_arxiv_pdfs(keywords: List[str], out_dir: str, max_results: int = 200, polite_delay: float = 0.25) -> List[str]:
    urls = fetch_arxiv_pdf_urls(keywords, max_results=max_results)
    entries = [{"url": u, "name": u.rstrip("/").split("/")[-1], "meta": {"source": "arxiv", "url": u}} for u in urls]
    paths = parallel_download_pdfs(entries, out_dir, max_workers=8)
    # small pause
    time.sleep(polite_delay)
    return paths


# --------------------------------------------------------------------------------------
# OpenAlex (robust, fallback strategies)
# --------------------------------------------------------------------------------------


def _openalex_build_search_query(keywords: List[str]) -> str:
    return " ".join(sorted(set(keywords), key=str.lower))


def _openalex_fetch_works_try(search: str, filter_str: str, per_page: int, page: int, mailto: Optional[str]) -> Dict[str, Any]:
    headers = {"User-Agent": f"polymer-rag/1.0 (+{mailto or DEFAULT_MAILTO})"}
    params = {
        "search": search,
        "per-page": per_page,
        "per_page": per_page,
        "page": page,
        "sort": "publication_date:desc",
    }
    if filter_str:
        params["filter"] = filter_str
    if mailto:
        params["mailto"] = mailto
    resp = requests.get(OPENALEX_WORKS_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _openalex_fetch_works(keywords: List[str], max_results: int = 2000, per_page: int = 200, mailto: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Try multiple query forms:
    - combined-space query
    - OR-joined query
    - single-keyword fallback
    Also retries with relaxed filters if needed.
    """
    kws = sorted(set(keywords or []), key=str.lower)
    # prepare query forms
    combined = " ".join(kws)
    or_query = " OR ".join(kws)
    singles = kws or ["polymer"]

    attempts = [
        {"q": combined, "filter": "is_oa:true,language:en"},
        {"q": or_query, "filter": "is_oa:true,language:en"},
        {"q": or_query, "filter": "is_oa:true"},
        {"q": or_query, "filter": ""},  # no filters
    ]
    # append single-key fallback attempts
    for s in singles[:3]:
        attempts.append({"q": s, "filter": ""})

    works: List[Dict[str, Any]] = []
    for attempt in attempts:
        search = attempt["q"]
        filter_str = attempt["filter"]
        page = 1
        # iterate pages
        while len(works) < max_results:
            try:
                data = _openalex_fetch_works_try(search, filter_str, per_page, page, mailto or DEFAULT_MAILTO)
            except Exception as e:
                print(f"[WARN] OpenAlex request failed for search='{search}' filter='{filter_str}': {e}")
                break
            results = data.get("results", [])
            print(f"[DEBUG] OpenAlex (search='{search[:120]}...' filter='{filter_str}') page={page} got {len(results)} results (total so far {len(works)})")
            if page == 1 and results:
                print("[DEBUG] sample result keys:", list(results[0].keys()))
            if not results:
                break
            works.extend(results)
            if len(results) < per_page:
                break
            page += 1
            time.sleep(0.12)
            if len(works) >= max_results:
                break
        if works:
            break
    # cap to max_results
    return works[:max_results]


def _openalex_extract_pdf_entries(works: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract candidate PDF URLs and name hints from OpenAlex works.
    Returns entries like {"url": pdf_url, "name": name, "meta": {...}}
    """
    out = []
    seen_urls = set()
    for w in works:
        pdf = ""
        # best_oa_location
        best = w.get("best_oa_location") or {}
        if isinstance(best, dict):
            pdf = best.get("pdf_url") or best.get("url_for_pdf") or best.get("url") or ""
        # primary_location
        if not pdf:
            pl = w.get("primary_location") or {}
            if isinstance(pl, dict):
                pdf = pl.get("pdf_url") or pl.get("url_for_pdf") or pl.get("landing_page_url") or ""
        # open_access fallback
        if not pdf:
            oa = w.get("open_access") or {}
            if isinstance(oa, dict):
                pdf = oa.get("oa_url") or oa.get("oa_url_for_pdf") or ""
        if not pdf:
            continue
        if pdf in seen_urls:
            continue
        seen_urls.add(pdf)
        title = (w.get("title") or w.get("display_name") or "").strip()
        year = w.get("publication_year") or w.get("publication_date") or ""
        venue = ""
        pl = w.get("primary_location") or {}
        if isinstance(pl, dict):
            venue = (pl.get("source") or {}).get("display_name") or ""
        if not venue:
            venue = ((w.get("host_venue") or {}).get("display_name") or "").strip()
        name = " - ".join([s for s in [title, venue, str(year or "")] if s])
        meta = {"title": title, "year": year, "venue": venue, "source": "openalex"}
        out.append({"url": pdf, "name": name, "meta": meta})
    return out


def fetch_openalex_pdfs(keywords: List[str], out_dir: str, max_results: int = 2000, per_page: int = 200, mailto: Optional[str] = None) -> List[str]:
    works = _openalex_fetch_works(keywords, max_results=max_results, per_page=per_page, mailto=mailto)
    if not works:
        print("[INFO] OpenAlex returned no works for given queries/filters.")
        return []
    entries = _openalex_extract_pdf_entries(works)
    if not entries:
        print("[INFO] OpenAlex works found, but no PDF links extracted.")
        return []
    print(f"[INFO] OpenAlex: {len(entries)} candidate PDF URLs extracted (will attempt download).")
    paths = parallel_download_pdfs(entries, out_dir, max_workers=16)
    return paths


# --------------------------------------------------------------------------------------
# Europe PMC fetching (additional OA source)
# --------------------------------------------------------------------------------------


def _epmc_query_from_keywords(keywords: List[str]) -> str:
    # build a simple AND/OR query that Europe PMC understands; keep compact
    q = " OR ".join([f'"{k}"' for k in keywords])
    return q


def _epmc_extract_pdf_entries_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for r in results:
        # Europe PMC 'fullTextUrlList' or 'fullTextUrl'
        ftl = r.get("fullTextUrlList") or {}
        urls = []
        # fullTextUrlList -> fullTextUrl is list of dicts with 'url' and 'documentStyle'
        if isinstance(ftl, dict):
            for ful in (ftl.get("fullTextUrl") or []):
                if isinstance(ful, dict):
                    u = ful.get("url") or ""
                    if u:
                        urls.append(u)
        # direct 'fullTextUrl' string
        if not urls:
            fu = r.get("fullTextUrl")
            if isinstance(fu, str) and fu:
                urls.append(fu)
        # also check 'doi' -> build DOI resolver landing page (not direct PDF) - skip for now
        for u in urls:
            if not u:
                continue
            if u in seen:
                continue
            seen.add(u)
            title = (r.get("title") or "").strip()
            year = r.get("firstPublicationDate") or r.get("pubYear") or ""
            name = " - ".join([s for s in [title, str(year or "")] if s])
            out.append({"url": u, "name": name, "meta": {"title": title, "year": year, "source": "epmc"}})
    return out


def fetch_epmc_pdfs(keywords: List[str], out_dir: str, max_results: int = 1000, page_size: int = 25) -> List[str]:
    """
    Query Europe PMC and extract fullTextUrlList entries. Europe PMC often contains links to
    PMC fulltext pages, publisher pages, or direct PDFs. We attempt all and let download_pdf filter for PDFs.
    """
    q = _epmc_query_from_keywords(keywords)
    params = {
        "query": q,
        "format": "json",
        "pageSize": page_size,
        "sort": "FIRST_PDATE_D desc",
    }
    headers = {"User-Agent": f"polymer-rag/1.0 (+{DEFAULT_MAILTO})"}
    saved = []
    cursor = 1
    total_fetched = 0
    while total_fetched < max_results:
        params["page"] = cursor
        try:
            resp = requests.get(EPMC_SEARCH_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[WARN] Europe PMC request failed: {e}")
            break
        results = data.get("resultList", {}).get("result", [])
        if not results:
            break
        entries = _epmc_extract_pdf_entries_from_results(results)
        if not entries:
            cursor += 1
            total_fetched += len(results)
            time.sleep(0.2)
            continue
        paths = parallel_download_pdfs(entries, out_dir, max_workers=8)
        saved.extend(paths)
        total_fetched += len(results)
        cursor += 1
        time.sleep(0.2)
    return saved


# --------------------------------------------------------------------------------------
# Embeddings: Smart wrapper for E5 prefixing
# --------------------------------------------------------------------------------------


class SmartHFEmbeddings(HuggingFaceEmbeddings):
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
# Local ensemble (RRF)
# --------------------------------------------------------------------------------------


class SimpleEnsembleRetriever:
    def __init__(self, retrievers, weights=None, k: int = 6, rrf_k: int = 60):
        assert retrievers, "At least one retriever required"
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        assert len(self.weights) == len(self.retrievers)
        self.k = k
        self.rrf_k = rrf_k

    def _run_retriever(self, retriever, query: str):
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
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
        all_lists = []
        for r in self.retrievers:
            docs = self._run_retriever(r, query)
            all_lists.append(docs or [])
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
# Builder: load PDFs, chunk, index (Chroma / FAISS)
# --------------------------------------------------------------------------------------


def _attach_extra_metadata_from_manifest(docs: List[Any], manifest: Dict[str, Dict[str, Any]]) -> None:
    for d in docs:
        src_path = d.metadata.get("source", "")  # some loaders store source path in metadata
        if not src_path:
            continue
        rec = manifest.get(src_path)
        if not rec:
            # try basename match
            for k, v in manifest.items():
                if os.path.basename(k) == os.path.basename(src_path):
                    rec = v
                    break
        if rec:
            for k in ("title", "year", "venue", "url", "source"):
                if k in rec:
                    d.metadata[k] = rec[k]


def _split_and_build_retriever(
    documents_dir: str,
    persist_dir: Optional[str] = None,
    k: int = 6,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    vector_backend: str = "chroma",
    min_chunk_chars: int = 200,
):
    print(f"🗂️  Loading PDFs from: {documents_dir}")
    loader = DirectoryLoader(documents_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
    docs = loader.load()
    if not docs:
        raise RuntimeError("No PDF documents found to index.")
    manifest = _load_manifest(documents_dir)
    _attach_extra_metadata_from_manifest(docs, manifest)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=250, length_function=len, separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(docs)
    documents = [d for d in documents if len(d.page_content or "") >= min_chunk_chars]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = max(k, 8)
    print(f"🔤 Using embeddings model: {embedding_model}")
    embeddings = SmartHFEmbeddings(model_name=embedding_model)
    if vector_backend.lower() == "chroma":
        if persist_dir:
            print(f"💾 Using Chroma persist_dir={persist_dir}")
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
            raise RuntimeError("FAISS requested but not available; pip install faiss-cpu") from e
        vector_store = FAISS.from_documents(documents, embeddings)
    else:
        raise ValueError("vector_backend must be 'chroma' or 'faiss'")
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    ensemble = SimpleEnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.45, 0.55], k=k)
    print("✅ RAG KB ready (BM25 + Vector ensemble).")
    return ensemble


# --------------------------------------------------------------------------------------
# High-level fetch builder that uses multiple sources and targets a large total
# --------------------------------------------------------------------------------------


def build_retriever_from_web(
    polymer_keywords: Optional[List[str]] = None,
    max_openalex: int = 3000,
    max_arxiv: int = 1000,
    max_epmc: int = 1000,
    max_total_pdfs: int = 5000,
    from_year: int = 2010,
    extra_pdf_urls: Optional[List[str]] = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    tmp_download_dir: str = DEFAULT_TMP_DOWNLOAD_DIR,
    k: int = 6,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    vector_backend: str = "chroma",
    mailto: Optional[str] = None,
):
    polymer_keywords = sorted(set(polymer_keywords or POLYMER_KEYWORDS), key=str.lower)
    print("📡 Fetching polymer PDFs from OpenAlex, arXiv, Europe PMC and extras...")
    _ensure_dir(tmp_download_dir)
    all_paths: List[str] = []
    seen_urls = set()

    # 1) OpenAlex (largest coverage) - fetch works then extract PDF links
    try:
        openalex_paths = fetch_openalex_pdfs(polymer_keywords, out_dir=tmp_download_dir, max_results=max_openalex, per_page=200, mailto=mailto)
        for p in openalex_paths:
            if p not in all_paths:
                all_paths.append(p)
    except Exception as e:
        print(f"[WARN] OpenAlex fetch error: {e}")

    # 2) arXiv (good specialized coverage)
    try:
        arxiv_paths = fetch_arxiv_pdfs(polymer_keywords, out_dir=tmp_download_dir, max_results=max_arxiv)
        for p in arxiv_paths:
            if p not in all_paths:
                all_paths.append(p)
    except Exception as e:
        print(f"[WARN] arXiv fetch error: {e}")

    # 3) Europe PMC
    try:
        epmc_paths = fetch_epmc_pdfs(polymer_keywords, out_dir=tmp_download_dir, max_results=max_epmc)
        for p in epmc_paths:
            if p not in all_paths:
                all_paths.append(p)
    except Exception as e:
        print(f"[WARN] Europe PMC fetch error: {e}")

    # 4) Extra URLs
    if extra_pdf_urls:
        extra_entries = [{"url": u, "name": None, "meta": {"url": u, "source": "extra"}} for u in extra_pdf_urls]
        extra_paths = parallel_download_pdfs(extra_entries, tmp_download_dir, max_workers=8)
        for p in extra_paths:
            if p not in all_paths:
                all_paths.append(p)

    # If not enough, attempt incremental fallback: try single-key searches and looser search forms
    total_found = len(all_paths)
    print(f"🔎 Initial fetched PDFs: {total_found}")
    if total_found < max_total_pdfs:
        print("[INFO] Not enough PDFs yet — attempting additional looser searches (OR-joined single-key fallbacks).")
        # Use single keywords to expand
        for kw in polymer_keywords:
            if len(all_paths) >= max_total_pdfs:
                break
            try:
                aa = fetch_openalex_pdfs([kw], out_dir=tmp_download_dir, max_results=200, per_page=200, mailto=mailto)
                for p in aa:
                    if p not in all_paths:
                        all_paths.append(p)
                time.sleep(0.12)
            except Exception:
                continue

    total = len(all_paths)
    print(f"✅ Downloaded {total} PDFs (OpenAlex/arXiv/EuropePMC/extra).")
    if total == 0:
        raise RuntimeError("No PDFs fetched. Adjust keywords or add extra_pdf_urls.")

    print("🧠 Building knowledge base from downloaded PDFs...")
    retriever = _split_and_build_retriever(documents_dir=tmp_download_dir, persist_dir=persist_dir, k=k, embedding_model=embedding_model, vector_backend=vector_backend)
    return retriever


# --------------------------------------------------------------------------------------
# Local builder from existing folder
# --------------------------------------------------------------------------------------


def build_retriever(
    papers_path: str,
    persist_dir: Optional[str] = DEFAULT_PERSIST_DIR,
    k: int = 6,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    vector_backend: str = "chroma",
):
    print("📚 Building RAG knowledge base from local PDFs...")
    return _split_and_build_retriever(documents_dir=papers_path, persist_dir=persist_dir, k=k, embedding_model=embedding_model, vector_backend=vector_backend)


# --------------------------------------------------------------------------------------
# Convenience wrapper
# --------------------------------------------------------------------------------------


def build_retriever_polymer_foundation_models(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    k: int = 6,
    from_year: int = 2015,
    vector_backend: str = "chroma",
):
    fm_kw = list(set(POLYMER_KEYWORDS + [
        "BigSMILES", "PSMILES", "polymer SMILES", "polymer language model",
        "foundation model polymer", "masked language model polymer",
        "self-supervised polymer", "generative polymer",
        "Perceiver polymer", "Performer polymer",
        "polymer sequence modeling", "representation learning polymer",
    ]))
    return build_retriever_from_web(polymer_keywords=fm_kw, max_openalex=4000, max_arxiv=800, max_epmc=800, max_total_pdfs=5000, from_year=from_year, persist_dir=persist_dir, k=k, vector_backend=vector_backend)


# --------------------------------------------------------------------------------------
# CLI smoke (example)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = build_retriever_from_web(
        polymer_keywords=POLYMER_KEYWORDS,
        max_openalex=2000,
        max_arxiv=500,
        max_epmc=500,
        max_total_pdfs=1200,
        persist_dir="chroma_polymer_db_big",
        tmp_download_dir=DEFAULT_TMP_DOWNLOAD_DIR,
        k=6,
        embedding_model="intfloat/e5-large-v2",
        vector_backend="chroma",
        mailto=DEFAULT_MAILTO,
    )
    print("🔎 Sample query:")
    docs = retriever.get_relevant_documents("PSMILES polymer electrolyte design")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        title = meta.get("title") or os.path.basename(meta.get("source", "")) or "document"
        year = meta.get("year", "")
        src = meta.get("source", "unknown")
        print(f"[{i}] {title} ({year}) [{src}] :: {(d.page_content or '')[:200]}...")
