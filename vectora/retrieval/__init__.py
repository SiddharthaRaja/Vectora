"""Hybrid retrieval: dense + lexical, fused by rank, reranked and diversified."""

from vectora.retrieval.diversify import cap_per_paper, deduplicate, mmr
from vectora.retrieval.expansion import Expansion, expand
from vectora.retrieval.fusion import RankedList, reciprocal_rank_fusion
from vectora.retrieval.pipeline import Retriever

__all__ = [
    "Expansion",
    "RankedList",
    "Retriever",
    "cap_per_paper",
    "deduplicate",
    "expand",
    "mmr",
    "reciprocal_rank_fusion",
]
