# agent_pipeline/tools/web_rag.py
"""
On-the-fly Web RAG for polymers.

- Uses Tavily (TAVILY_API_KEY) or SerpAPI (SERPAPI_API_KEY) for web search.
- Fetches top-N pages, strips HTML, chunks, embeds with OpenAI text-embedding-3-large.
- Ranks chunks by cosine similarity to the user query and returns concise evidence packs.

Env:
  OPENAI_API_KEY
  OPENAI_ORG (optional)
  TAVILY_API_KEY (preferred) or SERPAPI_API_KEY
  OPENAI_EMBED_MODEL (optional, default: text-embedding-3-large)
"""

from __future__ import annotations
import os, json, time, math, re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import hashlib
import threading

import requests
import numpy as np

LOGGER = logging.getLogger(__name__)

# --------- Utilities

def _clean_text(html: str) -> str:
    # Minimal HTML -> text
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _chunk(text: str, max_tokens: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, max_tokens - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i+max_tokens])
        chunks.append(chunk)
        i += step
    return chunks

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

# --------- Embeddings

class OpenAIEmbedder:
    def __init__(self, model: str | None = None, timeout: int = 60):
        self.model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.org = os.getenv("OPENAI_ORG")
        self.timeout = timeout

    def embed(self, texts: List[str]) -> np.ndarray:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.org:
            headers["OpenAI-Organization"] = self.org
        data = {"model": self.model, "input": texts}
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=data,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        arr = [d["embedding"] for d in resp.json()["data"]]
        return np.array(arr, dtype=np.float32)

# --------- Search backends

def _search_tavily(query: str, k: int = 5) -> List[Dict]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    r = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": k},
        timeout=30,
    )
    r.raise_for_status()
    js = r.json()
    results = []
    for item in js.get("results", []):
        results.append({"title": item.get("title"), "url": item.get("url")})
    return results

def _search_serpapi(query: str, k: int = 5) -> List[Dict]:
    key = os.getenv("SERPAPI_API_KEY")
    if not key:
        return []
    params = {"api_key": key, "engine": "google", "q": query, "num": k}
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    results = []
    for item in js.get("organic_results", [])[:k]:
        results.append({"title": item.get("title"), "url": item.get("link")})
    return results

def _fetch_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "PolymerAgent/1.0"})
        if r.status_code == 200 and r.text:
            return _clean_text(r.text)
    except Exception as e:
        LOGGER.warning("Fetch failed: %s (%s)", url, e)
    return ""

# --------- Main tool

@dataclass
class Evidence:
    title: str
    url: str
    snippet: str
    score: float

class WebRAGTool:
    """
    Search -> fetch -> chunk -> embed -> rank -> return top evidence chunks.
    """

    def __init__(self, max_results: int = 6, chunks_per_page: int = 4):
        self.max_results = max_results
        self.chunks_per_page = chunks_per_page
        self.embedder = OpenAIEmbedder()

    def search(self, query: str) -> List[Dict]:
        res = _search_tavily(query, self.max_results)
        if not res:
            res = _search_serpapi(query, self.max_results)
        return res

    def retrieve(self, query: str, query_expansion: Optional[str] = None, top_k: int = 6) -> List[Evidence]:
        q = query if not query_expansion else f"{query}. {query_expansion}"
        search_hits = self.search(q)
        if not search_hits:
            return []

        # Fetch pages concurrently
        texts: Dict[str, str] = {}
        threads = []
        def worker(u):
            texts[u] = _fetch_url(u)
        for hit in search_hits:
            t = threading.Thread(target=worker, args=(hit["url"],))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # Build corpus
        corpus = []
        mapping = []  # (title, url, chunk_text)
        for hit in search_hits:
            title, url = hit["title"] or "", hit["url"]
            text = texts.get(url, "")
            if not text:
                continue
            for chunk in _chunk(text)[: self.chunks_per_page]:
                corpus.append(chunk)
                mapping.append((title, url, chunk))

        if not corpus:
            return []

        # Rank
        qv = self.embedder.embed([q])[0]
        cv = self.embedder.embed(corpus)
        scores = [ _cosine(qv, cv[i]) for i in range(len(corpus)) ]
        idx = np.argsort(scores)[::-1][:top_k]

        out: List[Evidence] = []
        for i in idx:
            title, url, chunk = mapping[i]
            snippet = (chunk[:600] + "…") if len(chunk) > 600 else chunk
            out.append(Evidence(title=title, url=url, snippet=snippet, score=float(scores[i])))
        return out

    # For tool registry
    def tool_search(self, args: Dict) -> List[Dict]:
        return self.search(args["query"])

    def tool_retrieve(self, args: Dict) -> List[Dict]:
        ev = self.retrieve(args["query"], args.get("query_expansion"), args.get("top_k", 6))
        return [e.__dict__ for e in ev]
