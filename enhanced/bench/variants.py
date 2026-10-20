"""Retrieval variants a benchmark can run.

A variant names a retriever class, its constructor kwargs, and optional env
overrides (feature flags read at import). Hypotheses add their treatment here
next to the baseline they are compared against, so both run from one commit.

Every variant retrieves to the same depth (``DEPTH``) so recall@k is
comparable across variants for every k <= DEPTH.
"""

DEPTH = 20

VARIANTS = {
    # Upstream default search type, LoCoMo report config (20/20/20).
    "hybrid": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {"chunks_top_k": DEPTH, "entities_top_k": 20, "facts_top_k": 20},
        "env": {},
    },
    # Single dense lane over chunks.
    "chunks": {
        "retriever": "cognee.modules.retrieval.chunks_retriever:ChunksRetriever",
        "kwargs": {"top_k": DEPTH},
        "env": {},
    },
    # Single lexical lane (BM25) over chunks.
    "bm25": {
        "retriever": "cognee.modules.retrieval.bm25_retriever:BM25ChunksRetriever",
        "kwargs": {"top_k": DEPTH},
        "env": {},
    },
}
