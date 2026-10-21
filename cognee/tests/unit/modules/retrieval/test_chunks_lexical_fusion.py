from types import SimpleNamespace

import pytest

from cognee.modules.retrieval import chunks_retriever as chunks_mod
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever


@pytest.mark.asyncio
async def test_fused_chunks_use_the_hybrid_chunk_lane_without_summaries(monkeypatch):
    seen = {}

    async def fake_retrieve_hybrid_chunks(**kwargs):
        seen.update(kwargs)
        return {"chunks": ["c1", "c2"], "chunk_summaries": {}}

    async def fake_engine():
        return SimpleNamespace(vector=object(), graph=object())

    async def no_conflicts(chunks, graph_engine=None):
        return None

    import cognee.modules.retrieval.hybrid.chunks as hybrid_chunks

    monkeypatch.setattr(hybrid_chunks, "retrieve_hybrid_chunks", fake_retrieve_hybrid_chunks)
    monkeypatch.setattr(chunks_mod, "get_unified_engine", fake_engine)
    monkeypatch.setattr(chunks_mod, "attach_chunk_conflicts", no_conflicts)

    retriever = ChunksRetriever(top_k=7, node_name=["p"], lexical_fusion=True)

    assert await retriever.get_retrieved_objects("q") == ["c1", "c2"]
    assert seen["chunks_top_k"] == 7
    assert seen["text_summaries_top_k"] == 0
    assert seen["lexical_lane"] is True
    assert seen["node_name"] == ["p"]


def test_lexical_fusion_is_off_by_default():
    assert ChunksRetriever().lexical_fusion is False
