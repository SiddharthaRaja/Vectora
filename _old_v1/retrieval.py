"""
Research Paper RAG — Step 3: Retrieval
=======================================
Hybrid search (dense + sparse BM25) → cross-encoder reranking → MMR diversification.
Each result carries full citation metadata.

Usage:
  # Interactive mode — type queries one by one
  python retrieval.py

  # Single query
  python retrieval.py --query "What methods were used for data augmentation?"

  # Filter by specific paper or year
  python retrieval.py --query "transformer architecture" --year 2023
  python retrieval.py --query "loss function" --paper-id attention_is_all_you_need

Dependencies (add to requirements.txt):
  rank-bm25>=0.2.2
  sentence-transformers>=2.6.0   # already installed
  qdrant-client>=1.9.0           # already installed
"""

import json
import math
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

QDRANT_URL        = "http://localhost:6333"
COLLECTION_NAME   = "research_papers"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"           # same model used during ingestion
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # lightweight cross-encoder

DENSE_TOP_K       = 20    # candidates fetched from vector search
FINAL_TOP_K       = 5     # results returned after reranking + MMR
MMR_LAMBDA        = 0.6   # 0 = max diversity, 1 = max relevance (0.6 is a good balance)

# ── Search mode profiles ──────────────────────
# Passed in as mode="fast"|"balanced"|"deep"
SEARCH_MODES = {
    "fast": {
        "dense_top_k":    8,
        "use_hyde":       False,
        "rewrite_n":      0,
        "use_reranker":   False,
        "use_mmr":        False,
    },
    "balanced": {
        "dense_top_k":    15,
        "use_hyde":       True,
        "rewrite_n":      1,
        "use_reranker":   True,
        "use_mmr":        True,
    },
    "deep": {
        "dense_top_k":    20,
        "use_hyde":       True,
        "rewrite_n":      2,
        "use_reranker":   True,
        "use_mmr":        True,
    },
}
DEFAULT_MODE = "balanced"


# HyDE config
HYDE_MODEL        = "gpt-4o-mini"   # model used to generate hypothetical answer
HYDE_MAX_TOKENS   = 200             # keep it short — we only need a paragraph
HYDE_ENABLED      = True            # set False to disable globally

# Query rewriting config
REWRITE_MODEL     = "gpt-4o-mini"
REWRITE_N         = 2               # number of rewritten variants (2-3 recommended)
REWRITE_ENABLED   = True            # set False to disable globally


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:    str
    paper_id:    str
    title:       str
    authors:     str
    year:        str
    doi:         str
    section:     str
    page:        int
    chunk_type:  str
    content:     str
    image_path:  str
    score:       float      # final score after reranking

    def citation(self) -> str:
        """Returns a short inline citation string."""
        author_short = self.authors.split(",")[0].strip() if self.authors else self.paper_id
        year = self.year or "n.d."
        return f"[{author_short} et al., {year} — {self.section}, p.{self.page}]"

    def to_context_block(self) -> str:
        """Formats chunk for inclusion in an LLM prompt."""
        lines = [
            f"SOURCE: {self.title}",
            f"AUTHORS: {self.authors}",
            f"YEAR: {self.year}  |  DOI: {self.doi}",
            f"SECTION: {self.section}  |  PAGE: {self.page}  |  TYPE: {self.chunk_type}",
            f"---",
            self.content,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Model loader (singleton — loaded once)
# ─────────────────────────────────────────────

_embedder: SentenceTransformer = None
_reranker: CrossEncoder = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info(f"Loading embedder: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        log.info(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


# ─────────────────────────────────────────────
# Step 1: Dense retrieval (vector search)
# ─────────────────────────────────────────────

def dense_search(
    client: QdrantClient,
    query: str,
    top_k: int = DENSE_TOP_K,
    filters: dict = None,
    query_vec: list[float] = None,   # pre-computed vector (e.g. from HyDE)
) -> list[dict]:
    """
    Embed the query and search Qdrant for nearest neighbours.
    Returns list of payload dicts with an added 'dense_score' field.
    If query_vec is provided, skips embedding and uses it directly.
    """
    if query_vec is None:
        embedder = get_embedder()
        query_vec = embedder.encode(query, normalize_embeddings=True).tolist()

    # Build optional Qdrant filter (e.g. filter by year or paper_id)
    qdrant_filter = None
    if filters:
        conditions = []
        for field, value in filters.items():
            conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
        qdrant_filter = Filter(must=conditions)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        query_filter=qdrant_filter,
    )

    candidates = []
    for r in response.points:
        payload = dict(r.payload)
        payload["dense_score"] = r.score
        candidates.append(payload)

    return candidates


# ─────────────────────────────────────────────
# Step 2: Sparse retrieval (BM25)
# ─────────────────────────────────────────────

def bm25_search(
    candidates: list[dict],
    query: str,
    top_k: int = DENSE_TOP_K,
) -> list[dict]:
    """
    Re-score the dense candidates with BM25 and return a merged score.
    BM25 is run over the candidate pool (not the whole collection) for speed.
    For a larger corpus you'd maintain a persistent BM25 index separately.
    """
    if not candidates:
        return []

    tokenized_corpus = [c["content"].lower().split() for c in candidates]
    tokenized_query  = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalise BM25 scores to [0, 1]
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    norm_bm25 = bm25_scores / max_bm25

    # Hybrid score: weighted average of dense (cosine) and sparse (BM25)
    alpha = 0.6   # weight for dense; (1-alpha) for BM25
    for i, candidate in enumerate(candidates):
        candidate["bm25_score"]   = float(norm_bm25[i])
        candidate["hybrid_score"] = (
            alpha * candidate["dense_score"] +
            (1 - alpha) * float(norm_bm25[i])
        )

    # Re-sort by hybrid score and take top_k
    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return candidates[:top_k]


# ─────────────────────────────────────────────
# Step 3: Cross-encoder reranking
# ─────────────────────────────────────────────

def rerank(candidates: list[dict], query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
    """
    Pass (query, passage) pairs through a cross-encoder.
    Much more accurate than bi-encoder similarity but slower —
    only run on the small candidate set (≤20), not the whole collection.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c["content"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


# ─────────────────────────────────────────────
# Step 4: MMR (Maximal Marginal Relevance)
# ─────────────────────────────────────────────

def mmr(candidates: list[dict], query: str, top_k: int = FINAL_TOP_K, mmr_lambda: float = None) -> list[dict]:
    """
    Diversify results so the final set covers different aspects
    rather than returning near-duplicate chunks from the same section.

    MMR score = lambda * relevance - (1 - lambda) * max_similarity_to_selected
    """
    if len(candidates) <= top_k:
        return candidates

    embedder = get_embedder()

    # Embed query and all candidates
    texts  = [query] + [c["content"] for c in candidates]
    embeds = embedder.encode(texts, normalize_embeddings=True)
    query_emb      = embeds[0]
    candidate_embs = embeds[1:]

    selected_indices = []
    remaining        = list(range(len(candidates)))

    while len(selected_indices) < top_k and remaining:
        best_idx   = None
        best_score = -float("inf")

        for idx in remaining:
            relevance = float(np.dot(candidate_embs[idx], query_emb))

            if not selected_indices:
                redundancy = 0.0
            else:
                sims_to_selected = [
                    float(np.dot(candidate_embs[idx], candidate_embs[s]))
                    for s in selected_indices
                ]
                redundancy = max(sims_to_selected)

            lam = mmr_lambda if mmr_lambda is not None else MMR_LAMBDA
            mmr_score = lam * relevance - (1 - lam) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


# ─────────────────────────────────────────────
# Step 5: Deduplication
# ─────────────────────────────────────────────

def deduplicate(candidates: list[dict]) -> list[dict]:
    """
    Remove chunks with identical chunk_id (same content from different papers).
    Also removes chunks where content is >90% similar to an already-selected chunk.
    """
    seen_ids   = set()
    seen_texts = []
    unique     = []

    embedder = get_embedder()

    for c in candidates:
        if c["chunk_id"] in seen_ids:
            continue

        # Check near-duplicate content
        if seen_texts:
            emb  = embedder.encode(c["content"], normalize_embeddings=True)
            sims = [float(np.dot(emb, s)) for s in seen_texts]
            if max(sims) > 0.92:
                continue   # near-duplicate — skip

        seen_ids.add(c["chunk_id"])
        seen_texts.append(embedder.encode(c["content"], normalize_embeddings=True))
        unique.append(c)

    return unique



# ─────────────────────────────────────────────
# HyDE — Hypothetical Document Embedding
# ─────────────────────────────────────────────

def generate_hypothetical_answer(query: str, openai_client: OpenAI) -> str:
    """
    Ask GPT-4o-mini to write a short hypothetical answer to the query.
    This answer is then embedded instead of (or alongside) the raw query,
    dramatically improving retrieval for short or vague questions.

    Example:
      Query:      "vitamin D deficiency methods"
      Hypothetical: "Vitamin D deficiency was assessed using serum 25(OH)D
                    concentration measured by ELISA. Deficiency was defined
                    as levels below 20 ng/mL per standard clinical guidelines..."
    """
    try:
        response = openai_client.chat.completions.create(
            model=HYDE_MODEL,
            max_tokens=HYDE_MAX_TOKENS,
            temperature=0.7,   # slight creativity helps cover more vocabulary
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a scientific writing assistant. When given a research question, "
                        "write a single dense paragraph (3-5 sentences) that looks like an excerpt "
                        "from a scientific paper answering that question. Use formal academic language "
                        "and domain-specific terminology. Do not mention that this is hypothetical."
                    )
                },
                {
                    "role": "user",
                    "content": f"Write a hypothetical research paper excerpt that answers: {query}"
                }
            ]
        )
        hypothetical = response.choices[0].message.content.strip()
        log.info(f"  HyDE generated: {hypothetical[:80]}...")
        return hypothetical
    except Exception as e:
        log.warning(f"  HyDE failed ({e}), falling back to raw query")
        return query   # graceful fallback — pipeline continues with original query


def hyde_embed(query: str, openai_client: OpenAI) -> list[float]:
    """
    Generate a hypothetical answer and return its embedding.
    The embedding is a blend of the query embedding and the
    hypothetical answer embedding for balanced retrieval.
    """
    embedder = get_embedder()

    # Embed the original query
    query_emb = embedder.encode(query, normalize_embeddings=True)

    if not HYDE_ENABLED or openai_client is None:
        return query_emb.tolist()

    # Generate and embed the hypothetical answer
    hypothetical = generate_hypothetical_answer(query, openai_client)
    hyde_emb = embedder.encode(hypothetical, normalize_embeddings=True)

    # Blend: 40% original query + 60% hypothetical
    # The blend ensures we don't drift too far from the actual question
    blended = 0.4 * query_emb + 0.6 * hyde_emb

    # Re-normalise after blending
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm

    return blended.tolist()



# ─────────────────────────────────────────────
# Query rewriting
# ─────────────────────────────────────────────

def rewrite_query(query: str, openai_client: OpenAI, n: int = REWRITE_N) -> list[str]:
    """
    Generate N alternative phrasings of the query using academic language.
    Returns a list of rewritten queries (does NOT include the original).

    Example:
      Input:  "vitamin D and bones"
      Output: [
        "effect of vitamin D supplementation on bone mineral density",
        "calcitriol role in skeletal metabolism and osteoporosis prevention"
      ]
    """
    try:
        response = openai_client.chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=200,
            temperature=0.6,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a scientific search assistant. Rewrite the given research question "
                        f"into {n} different phrasings that use formal academic and domain-specific "
                        "terminology. Each variant should approach the same topic from a slightly "
                        "different angle to maximise retrieval coverage. "
                        f"Return exactly {n} variants, one per line, no numbering or bullets."
                    )
                },
                {
                    "role": "user",
                    "content": f"Rewrite this query: {query}"
                }
            ]
        )
        raw = response.choices[0].message.content.strip()
        variants = [line.strip() for line in raw.split("\n") if line.strip()][:n]
        log.info(f"  Query rewrites: {variants}")
        return variants
    except Exception as e:
        log.warning(f"  Query rewriting failed ({e}), using original only")
        return []


def retrieve_multi(
    query: str,
    client: QdrantClient,
    top_k: int = DENSE_TOP_K,
    filters: dict = None,
    openai_client: OpenAI = None,
) -> list[dict]:
    """
    Run dense_search for the original query + each rewritten variant.
    Merge all candidate pools, keeping the best score per unique chunk_id.
    Returns a flat deduplicated list of candidate dicts.
    """
    queries_to_run = [query]

    if REWRITE_ENABLED and openai_client is not None:
        variants = rewrite_query(query, openai_client)
        queries_to_run.extend(variants)

    log.info(f"  Running retrieval for {len(queries_to_run)} query variant(s)")

    # Collect candidates from all queries, keyed by chunk_id
    merged: dict[str, dict] = {}

    for q in queries_to_run:
        # HyDE embed each variant
        if HYDE_ENABLED and openai_client is not None:
            q_vec = hyde_embed(q, openai_client)
        else:
            q_vec = None

        candidates = dense_search(client, q, top_k=top_k, filters=filters, query_vec=q_vec)

        for c in candidates:
            cid = c.get("chunk_id", "")
            # Keep the highest dense_score if this chunk appears in multiple variant results
            if cid not in merged or c["dense_score"] > merged[cid]["dense_score"]:
                merged[cid] = c

    combined = list(merged.values())
    log.info(f"  Merged pool     → {len(combined)} unique candidates")
    return combined


# ─────────────────────────────────────────────
# Full retrieval pipeline
# ─────────────────────────────────────────────


def retrieve(
    query: str,
    client: QdrantClient,
    top_k: int = FINAL_TOP_K,
    filters: dict = None,
    openai_client: OpenAI = None,
    route_config: dict = None,
    mode: str = DEFAULT_MODE,
) -> list[RetrievedChunk]:
    """
    Full pipeline with mode-gated stages:

    fast     — dense search only, no HyDE/rewriting/reranking (~3-4s)
    balanced — HyDE + 1 rewrite + reranker + MMR (~6-8s)
    deep     — HyDE + 2 rewrites + reranker + MMR + larger pool (~10-14s)

    route_config keys (all optional, override mode defaults):
      top_k          -- override top_k
      mmr_lambda     -- override MMR_LAMBDA
      section_filter -- restrict to a section name
    """
    log.info(f"Query: '{query}' [mode={mode}]")

    # Resolve mode profile
    profile    = SEARCH_MODES.get(mode, SEARCH_MODES[DEFAULT_MODE])
    rc         = route_config or {}
    effective_k    = rc.get("top_k", top_k)
    mmr_lam        = rc.get("mmr_lambda", None)
    dense_k        = profile["dense_top_k"]

    effective_filters = dict(filters or {})
    if rc.get("section_filter"):
        effective_filters["section"] = rc["section_filter"]
    if not effective_filters:
        effective_filters = None

    log.info(f"  Profile: hyde={profile['use_hyde']}, "
             f"rewrites={profile['rewrite_n']}, "
             f"reranker={profile['use_reranker']}, "
             f"mmr={profile['use_mmr']}")

    # 1. Retrieval — mode-gated HyDE + rewriting
    if mode == "fast" or not profile["use_hyde"] and profile["rewrite_n"] == 0:
        # Fast path: plain dense search, no HyDE, no rewriting
        query_vec  = None   # dense_search will embed raw query
        candidates = dense_search(client, query, top_k=dense_k,
                                  filters=effective_filters, query_vec=query_vec)
        log.info(f"  Fast dense     -> {len(candidates)} candidates")
    else:
        # Balanced/Deep: use retrieve_multi with mode-controlled rewrite count
        original_rewrite_n = REWRITE_N
        import retrieval as _self
        _self.REWRITE_N = profile["rewrite_n"]
        candidates = retrieve_multi(query, client, top_k=dense_k,
                                    filters=effective_filters,
                                    openai_client=openai_client if profile["use_hyde"] else None)
        _self.REWRITE_N = original_rewrite_n  # restore

    # 2. BM25 hybrid re-score (always on — cheap)
    candidates = bm25_search(candidates, query, top_k=dense_k)
    log.info(f"  BM25 hybrid    -> {len(candidates)} candidates")

    # 3. Cross-encoder rerank (mode-gated)
    if profile["use_reranker"]:
        candidates = rerank(candidates, query, top_k=min(effective_k * 3, dense_k))
        log.info(f"  Reranked       -> {len(candidates)} candidates")
    else:
        # Fast path: just take top-k by hybrid score
        candidates = candidates[:effective_k]
        log.info(f"  Skipped rerank -> {len(candidates)} candidates (fast mode)")

    # 4. MMR (mode-gated)
    if profile["use_mmr"]:
        candidates = mmr(candidates, query, top_k=effective_k, mmr_lambda=mmr_lam)
        log.info(f"  MMR diversify  -> {len(candidates)} candidates")

    # 5. Deduplication
    candidates = deduplicate(candidates)
    log.info(f"  Deduplicated   -> {len(candidates)} final chunks")

    # Convert to RetrievedChunk dataclass
    results = []
    for c in candidates:
        results.append(RetrievedChunk(
            chunk_id   = c.get("chunk_id", ""),
            paper_id   = c.get("paper_id", ""),
            title      = c.get("title", ""),
            authors    = c.get("authors", ""),
            year       = c.get("year", ""),
            doi        = c.get("doi", ""),
            section    = c.get("section", ""),
            page       = c.get("page", 0),
            chunk_type = c.get("chunk_type", "text"),
            content    = c.get("content", ""),
            image_path = c.get("image_path", ""),
            score      = c.get("rerank_score", c.get("hybrid_score", 0.0)),
        ))

    return results


# ─────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────

def print_results(results: list[RetrievedChunk], query: str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.citation()}")
        print(f"    Score  : {r.score:.4f}")
        print(f"    Type   : {r.chunk_type}")
        print(f"    Content: {r.content[:250]}{'...' if len(r.content) > 250 else ''}")
    print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval — hybrid search + rerank + MMR")
    parser.add_argument("--query",     type=str, help="Search query (omit for interactive mode)")
    parser.add_argument("--top-k",     type=int, default=FINAL_TOP_K, help="Number of results")
    parser.add_argument("--year",      type=str, help="Filter by publication year")
    parser.add_argument("--paper-id",  type=str, help="Filter by paper ID (PDF filename stem)")
    parser.add_argument("--in-memory", action="store_true",
                        help="Use in-memory Qdrant (must have embedded with --in-memory too)")
    args = parser.parse_args()

    # Connect to Qdrant
    client = QdrantClient(":memory:") if args.in_memory else QdrantClient(url=QDRANT_URL)

    # Build filters
    filters = {}
    if args.year:
        filters["year"] = args.year
    if args.paper_id:
        filters["paper_id"] = args.paper_id

    if args.query:
        # Single query mode
        results = retrieve(args.query, client, top_k=args.top_k,
                           filters=filters or None)
        print_results(results, args.query)
    else:
        # Interactive mode
        print("\nRAG Retrieval — interactive mode  (type 'quit' to exit)\n")
        while True:
            try:
                query = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            results = retrieve(query, client, top_k=args.top_k,
                               filters=filters or None)
            print_results(results, query)


if __name__ == "__main__":
    main()
