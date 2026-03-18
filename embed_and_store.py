"""
Research Paper RAG — Step 2: Embed + Store in Qdrant
======================================================
Reads chunks.jsonl produced by ingestion_pipeline.py,
embeds each chunk with sentence-transformers, and uploads
vectors + metadata into a Qdrant collection.

Usage:
  python embed_and_store.py --input ./chunks.jsonl --collection research_papers

  # In-memory mode (no Docker needed, data lost on exit — good for testing):
  python embed_and_store.py --input ./chunks.jsonl --in-memory

Dependencies (add to requirements.txt):
  qdrant-client>=1.9.0
  sentence-transformers>=2.6.0
  tqdm>=4.66.0
"""

import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
VECTOR_DIM       = 384                   # must match the model above
BATCH_SIZE       = 64                    # chunks embedded per batch
UPLOAD_BATCH     = 128                   # points uploaded to Qdrant per batch
QDRANT_URL       = "http://localhost:6333"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_chunks(jsonl_path: Path, since_paper_ids: set = None) -> list[dict]:
    """
    Load chunks from JSONL.
    If since_paper_ids is provided, only return chunks from those papers
    (used for incremental embedding of new papers only).
    """
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if since_paper_ids is None or chunk.get("paper_id") in since_paper_ids:
                chunks.append(chunk)
    log.info(f"Loaded {len(chunks)} chunks from {jsonl_path}")
    return chunks


def get_existing_point_ids(client: QdrantClient, collection_name: str) -> set:
    """Return set of paper_ids already in the Qdrant collection."""
    existing = set()
    try:
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=["paper_id"],
                with_vectors=False,
            )
            for r in results:
                pid = (r.payload or {}).get("paper_id")
                if pid:
                    existing.add(pid)
            if next_offset is None:
                break
            offset = next_offset
    except Exception:
        pass  # collection may not exist yet
    return existing


def get_qdrant_client(in_memory: bool) -> QdrantClient:
    if in_memory:
        log.info("Using in-memory Qdrant (data will not persist)")
        return QdrantClient(":memory:")
    log.info(f"Connecting to Qdrant at {QDRANT_URL}")
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient, collection_name: str):
    """Create the collection if it doesn't exist yet."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        log.info(f"Collection '{collection_name}' already exists — will upsert into it")
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        # Create payload indexes for fast metadata filtering
        for field, schema in [
            ("paper_id",    PayloadSchemaType.KEYWORD),
            ("chunk_type",  PayloadSchemaType.KEYWORD),
            ("section",     PayloadSchemaType.KEYWORD),
            ("year",        PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
            )
        log.info(f"Created collection '{collection_name}' with {VECTOR_DIM}-dim vectors")


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Embed all chunks in batches. Returns list of embedding vectors."""
    log.info(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks (batch size {BATCH_SIZE})...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity = dot product after normalisation
    )
    return embeddings.tolist()


def upload_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    embeddings: list[list[float]],
):
    """
    Upload vectors + payloads in batches.
    The payload (metadata) stored alongside each vector:
      chunk_id, paper_id, title, authors, year, doi,
      section, page, chunk_type, content, image_path, token_estimate
    """
    log.info(f"Uploading {len(chunks)} points to Qdrant...")

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=i,                    # integer ID (Qdrant requires int or UUID)
            vector=vector,
            payload={                # everything stored here is searchable/filterable
                "chunk_id":       chunk["chunk_id"],
                "paper_id":       chunk["paper_id"],
                "title":          chunk["title"],
                "authors":        chunk["authors"],
                "year":           chunk["year"],
                "doi":            chunk["doi"],
                "section":        chunk["section"],
                "page":           chunk["page"],
                "chunk_type":     chunk["chunk_type"],
                "content":        chunk["content"],
                "image_path":     chunk["image_path"],
                "token_estimate": chunk["token_estimate"],
            },
        ))

    # Upload in batches
    for start in tqdm(range(0, len(points), UPLOAD_BATCH), desc="Uploading"):
        batch = points[start : start + UPLOAD_BATCH]
        client.upsert(collection_name=collection_name, points=batch)

    log.info("Upload complete.")


def print_collection_stats(client: QdrantClient, collection_name: str):
    info = client.get_collection(collection_name)
    count = client.count(collection_name).count
    print(f"\n{'='*50}")
    print(f"Collection : {collection_name}")
    print(f"Vectors    : {count}")
    print(f"Dimensions : {info.config.params.vectors.size}")
    print(f"Distance   : {info.config.params.vectors.distance}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# Quick sanity-check query
# ─────────────────────────────────────────────

def sanity_check(client: QdrantClient, collection_name: str):
    """Run a test query to confirm retrieval is working."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    test_query = "What methods were used in this study?"
    log.info(f"\nSanity check — querying: '{test_query}'")

    query_vec = model.encode(test_query, normalize_embeddings=True).tolist()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=3,
        with_payload=True,
    )
    results = response.points

    print("\nTop 3 results:")
    for i, r in enumerate(results, 1):
        p = r.payload
        print(f"\n  [{i}] score={r.score:.3f}")
        print(f"      Paper   : {p['title'][:60]}")
        print(f"      Section : {p['section']}")
        print(f"      Page    : {p['page']}")
        print(f"      Type    : {p['chunk_type']}")
        print(f"      Content : {p['content'][:150]}...")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Embed chunks and store in Qdrant (incremental)"
    )
    parser.add_argument("--input",      default="./chunks.jsonl",   help="Path to chunks.jsonl")
    parser.add_argument("--collection", default="research_papers",  help="Qdrant collection name")
    parser.add_argument("--in-memory",  action="store_true",        help="Use in-memory Qdrant")
    parser.add_argument("--sanity-check", action="store_true",      help="Run a test query after uploading")
    parser.add_argument("--force",      action="store_true",
                        help="Re-embed ALL chunks, even already-uploaded ones")
    args = parser.parse_args()

    # 1. Connect / create collection
    client = get_qdrant_client(args.in_memory)
    ensure_collection(client, args.collection)

    # 2. Find which paper_ids are already in Qdrant
    if args.force:
        already_embedded = set()
        log.info("--force: re-embedding all chunks")
    else:
        already_embedded = get_existing_point_ids(client, args.collection)
        log.info(f"Already in Qdrant: {len(already_embedded)} paper(s) — will skip these")

    # 3. Load only chunks from new papers
    all_chunks = load_chunks(Path(args.input))
    if not all_chunks:
        log.error("No chunks found. Run ingestion_pipeline.py first.")
        return

    new_paper_ids = {c["paper_id"] for c in all_chunks} - already_embedded
    if not new_paper_ids:
        log.info("\n✓ Nothing to embed — all papers already in Qdrant.")
        log.info("   Run ingestion_pipeline.py with new PDFs first, then re-run this script.")
        print_collection_stats(client, args.collection)
        return

    log.info(f"New papers to embed: {sorted(new_paper_ids)}")
    chunks = [c for c in all_chunks if c["paper_id"] in new_paper_ids]
    log.info(f"Chunks to embed: {len(chunks)}")

    # 4. Embed
    embeddings = embed_chunks(chunks)

    # 5. Upload (upsert — safe to run multiple times)
    upload_to_qdrant(client, args.collection, chunks, embeddings)

    # 6. Stats
    print_collection_stats(client, args.collection)

    # 7. Optional sanity check
    if args.sanity_check:
        sanity_check(client, args.collection)


if __name__ == "__main__":
    main()
