"""Optional cross-encoder rerank of the hybrid chunk lane (enhanced/HYPOTHESES.md H3).

The chunk lane fuses to a pool of ``rerank_top_n`` candidates, a local
fastembed cross-encoder scores each (query, chunk text) pair, and the pool is
reordered by that score before the usual cut to ``chunks_top_k``. Models load
once per process; any failure keeps the fused order.
"""

import asyncio
from typing import Any

from cognee.modules.retrieval.utils.results import display_value, payload
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")

_ENCODERS: dict[str, Any] = {}


def _encoder(model_name: str):
    if model_name not in _ENCODERS:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _ENCODERS[model_name] = TextCrossEncoder(model_name=model_name)
    return _ENCODERS[model_name]


def _score(model_name: str, query: str, texts: list[str]) -> list[float]:
    return list(_encoder(model_name).rerank(query, texts))


async def rerank_chunks(model_name: str, query: str, chunks: list[Any]) -> list[Any]:
    """``chunks`` reordered by cross-encoder score, highest first; ties keep fused order."""
    if len(chunks) < 2:
        return chunks
    texts = [display_value(payload(chunk).get("text")) or "" for chunk in chunks]
    try:
        scores = await asyncio.to_thread(_score, model_name, query, texts)
    except Exception:
        logger.warning("Hybrid rerank failed; keeping the fused order", exc_info=True)
        return chunks
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [chunks[i] for i in order]
