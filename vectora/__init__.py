"""
Vectora — retrieval-augmented question answering over research papers.

Four stages, each in its own subpackage:

* :mod:`vectora.ingestion` — PDFs to clean, metadata-rich chunks
* :mod:`vectora.store`     — dense vectors in Qdrant, lexical postings in BM25
* :mod:`vectora.retrieval` — hybrid search, rank fusion, reranking, diversification
* :mod:`vectora.generation`— routing, answering, citation resolution, grounding

:mod:`vectora.config` holds every tunable; nothing else reads the
environment.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
