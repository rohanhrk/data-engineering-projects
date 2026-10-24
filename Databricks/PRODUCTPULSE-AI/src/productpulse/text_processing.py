"""Text cleaning and chunking utilities for the RAG corpus.

The goal is not to create a fancy NLP framework. We want predictable chunks
that preserve enough retail context for citations: product, channel, date,
issue tags, and source type.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


def keyword_score(query: str, text: str) -> float:
    """Small lexical scoring helper used by the transparent reranker.

    Lakebase vector retrieval gives us semantic similarity search. This helper
    provides a simple additional lexical signal when teaching reranking without
    another service dependency.
    """

    # Extract simple alphanumeric tokens from both strings. This intentionally
    # avoids stemming or NLP libraries so students can see the reranking signal.
    query_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", query)}
    text_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", text)}
    if not query_terms:
        return 0.0
    
    # Score is the fraction of query terms that appear in the candidate text.
    # Example: if the query has 4 terms and 2 appear in the chunk, score = 0.5.
    return len(query_terms & text_terms) / len(query_terms)