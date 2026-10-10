"""
Quick smoke test — run this on a single PDF to verify the pipeline works.

Usage:
  python test_ingestion.py path/to/paper.pdf
"""

import sys
import json
from pathlib import Path

# Add parent dir to path if running from this folder
sys.path.insert(0, str(Path(__file__).parent))
from ingestion_pipeline import process_pdf, CAPTION_IMAGES


def test_single(pdf_path: str):
    path = Path(pdf_path)
    if not path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    seen = set()
    chunks = process_pdf(path, seen)

    print(f"\n{'='*60}")
    print(f"PDF: {path.name}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*60}\n")

    # Count by type
    from collections import Counter
    type_counts = Counter(c.chunk_type for c in chunks)
    print("Chunk types:")
    for k, v in type_counts.items():
        print(f"  {k:10s}  {v}")

    # Show sections found
    sections = list(dict.fromkeys(c.section for c in chunks))
    print(f"\nSections detected ({len(sections)}):")
    for s in sections:
        print(f"  • {s}")

    # Print 3 sample text chunks
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    print(f"\nSample text chunks (showing 3 of {len(text_chunks)}):")
    for c in text_chunks[:3]:
        print(f"\n  [Section: {c.section}  |  Page: {c.page}  |  ~{c.token_estimate} tokens]")
        preview = c.content[:280].replace("\n", " ")
        print(f"  {preview}{'...' if len(c.content) > 280 else ''}")

    # Print sample table chunk if any
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    if table_chunks:
        print(f"\nSample table chunk:")
        print(table_chunks[0].content[:400])

    # Print image chunks
    image_chunks = [c for c in chunks if c.chunk_type == "image"]
    if image_chunks:
        print(f"\nImage chunks ({len(image_chunks)}):")
        for c in image_chunks[:3]:
            print(f"  {c.image_path}")
            print(f"  Caption: {c.content}")

    # Dump first chunk as full JSON
    print("\nFull JSON for first chunk:")
    from dataclasses import asdict
    print(json.dumps(asdict(chunks[0]), indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ingestion.py path/to/paper.pdf")
        sys.exit(1)
    test_single(sys.argv[1])
