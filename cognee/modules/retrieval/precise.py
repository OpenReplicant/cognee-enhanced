"""Opt-in precise retrieval: cross-encoder reranking on the default retrieval paths.

Precise mode turns on the configuration benchmarked as
``hybrid_bm25_rr_minilm`` (enhanced/HYPOTHESES.md H3): the BM25 lane joins the
chunk fusion, the fused chunks are pooled to 30, and a local MiniLM
cross-encoder reorders the pool before the cut. On held-out LoCoMo questions
it raised recall@5 by 8.6 pt and MRR by 11.3 pt over the same pool without
reranking, and on LongMemEval-S MRR by 6.3 pt, at a cost of about 1.9 s per
query on a 4-core CPU (more on the first call, which downloads the 80 MB
model). That cost is why it is opt-in.

Enable per call with ``retriever_specific_config={"precise": True}`` or for the
process with ``COGNEE_PRECISE_RETRIEVAL=true``; an explicit per-call value
wins over the environment.
"""

import os

PRECISE_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
PRECISE_RERANK_TOP_N = 30


def precise_retrieval_enabled(retriever_specific_config: dict | None = None) -> bool:
    config = retriever_specific_config or {}
    if "precise" in config:
        return bool(config["precise"])
    value = os.getenv("COGNEE_PRECISE_RETRIEVAL", "false").strip().lower()
    return value in ("1", "true", "yes", "on")


def precise_rerank_kwargs(enabled: bool) -> dict:
    """Retriever kwargs that turn precise mode on; empty when it is off."""
    if not enabled:
        return {}
    return {"rerank_top_n": PRECISE_RERANK_TOP_N, "rerank_model": PRECISE_RERANK_MODEL}
