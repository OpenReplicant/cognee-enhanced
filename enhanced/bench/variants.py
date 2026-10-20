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
    # H1: hybrid + BM25 rank as a third chunk-lane RRF input.
    "hybrid_bm25": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "lexical_lane": True,
        },
        "env": {},
    },
    # H2: H1 with lane weights tuned on haiku dev convs 0-4
    # (results/locomo/H2-tune-haiku.json): dense chunk 0, dense summary 0.25, lexical 1.
    "hybrid_bm25_w_haiku": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "lexical_lane": True,
            "chunk_lane_weights": [0.0, 0.25, 1.0],
        },
        "env": {},
    },
    # H2b: H1 with lane weights tuned on haiku dev convs 0-4 under "no category
    # drops > 1 pt" (results/locomo/H2-tune-haiku-maxdrop0.01.json).
    "hybrid_bm25_w2_haiku": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "lexical_lane": True,
            "chunk_lane_weights": [0.5, 0.25, 1.0],
        },
        "env": {},
    },
    # H6: hybrid + Personalized PageRank chunk lane (fourth RRF input).
    "hybrid_ppr": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "ppr_lane": True,
        },
        "env": {},
    },
    # H6 on top of H1: dense chunk + dense summary + BM25 + PPR.
    "hybrid_bm25_ppr": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "lexical_lane": True,
            "ppr_lane": True,
        },
        "env": {},
    },
    # Ablation: hybrid without the dense summary lane.
    "hybrid_nosum": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "text_summaries_top_k": 0,
        },
        "env": {},
    },
    # Ablation: H1 without the dense summary lane (dense chunk + BM25 only).
    "hybrid_bm25_nosum": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "text_summaries_top_k": 0,
            "lexical_lane": True,
        },
        "env": {},
    },
    # H3 control: H1 fused to a pool of 30, cut to 20, no rerank.
    "hybrid_bm25_pool30": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "lexical_lane": True,
            "rerank_top_n": 30,
        },
        "env": {},
    },
    # H3: H1 pool of 30 reranked by MiniLM-L-6 cross-encoder (80 MB).
    "hybrid_bm25_rr_minilm": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "lexical_lane": True,
            "rerank_top_n": 30,
            "rerank_model": "Xenova/ms-marco-MiniLM-L-6-v2",
        },
        "env": {},
    },
    # H3: H1 pool of 30 reranked by bge-reranker-base (1 GB).
    "hybrid_bm25_rr_bge": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": DEPTH,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "lexical_lane": True,
            "rerank_top_n": 30,
            "rerank_model": "BAAI/bge-reranker-base",
        },
        "env": {},
    },
    # H4a: five passages instead of twenty; entity and fact lanes unchanged.
    "hybrid_k5": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {"chunks_top_k": 5, "entities_top_k": 20, "facts_top_k": 20},
        "env": {},
    },
    "hybrid_bm25_k5": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": 5,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "lexical_lane": True,
        },
        "env": {},
    },
    "hybrid_bm25_rr_minilm_k5": {
        "retriever": "cognee.modules.retrieval.hybrid_retriever:HybridRetriever",
        "kwargs": {
            "chunks_top_k": 5,
            "entities_top_k": 20,
            "facts_top_k": 20,
            "lexical_lane": True,
            "rerank_top_n": 30,
            "rerank_model": "Xenova/ms-marco-MiniLM-L-6-v2",
        },
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
