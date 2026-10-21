from types import SimpleNamespace

import pytest

from cognee.modules.retrieval import chunks_retriever as chunks_mod
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.precise import (
    PRECISE_RERANK_MODEL,
    PRECISE_RERANK_TOP_N,
    precise_rerank_kwargs,
    precise_retrieval_enabled,
)


def test_precise_is_off_by_default(monkeypatch):
    monkeypatch.delenv("COGNEE_PRECISE_RETRIEVAL", raising=False)
    assert precise_retrieval_enabled() is False
    assert precise_retrieval_enabled({}) is False
    assert precise_rerank_kwargs(False) == {}


@pytest.mark.parametrize("value", ["true", "1", "YES", " on "])
def test_env_turns_precise_on(monkeypatch, value):
    monkeypatch.setenv("COGNEE_PRECISE_RETRIEVAL", value)
    assert precise_retrieval_enabled() is True


def test_per_call_value_wins_over_env(monkeypatch):
    monkeypatch.setenv("COGNEE_PRECISE_RETRIEVAL", "true")
    assert precise_retrieval_enabled({"precise": False}) is False
    monkeypatch.setenv("COGNEE_PRECISE_RETRIEVAL", "false")
    assert precise_retrieval_enabled({"precise": True}) is True


def test_precise_kwargs_name_the_benchmarked_reranker():
    assert precise_rerank_kwargs(True) == {
        "rerank_top_n": PRECISE_RERANK_TOP_N,
        "rerank_model": PRECISE_RERANK_MODEL,
    }


@pytest.mark.asyncio
async def test_fused_chunks_pool_rerank_then_cut(monkeypatch):
    seen = {}

    async def fake_retrieve_hybrid_chunks(**kwargs):
        seen.update(kwargs)
        return {"chunks": [f"c{i}" for i in range(30)], "chunk_summaries": {}}

    async def fake_rerank(model_name, query, chunks, top_n=None, max_chars=None):
        seen["rerank"] = (model_name, query, top_n)
        return list(reversed(chunks))

    async def fake_engine():
        return SimpleNamespace(vector=object(), graph=object())

    async def no_conflicts(chunks, graph_engine=None):
        return None

    import cognee.modules.retrieval.hybrid.chunks as hybrid_chunks
    import cognee.modules.retrieval.hybrid.rerank as rerank_mod

    monkeypatch.setattr(hybrid_chunks, "retrieve_hybrid_chunks", fake_retrieve_hybrid_chunks)
    monkeypatch.setattr(rerank_mod, "rerank_chunks", fake_rerank)
    monkeypatch.setattr(chunks_mod, "get_unified_engine", fake_engine)
    monkeypatch.setattr(chunks_mod, "attach_chunk_conflicts", no_conflicts)

    retriever = ChunksRetriever(top_k=5, lexical_fusion=True, rerank_top_n=30, rerank_model="m")

    assert await retriever.get_retrieved_objects("q") == ["c29", "c28", "c27", "c26", "c25"]
    assert seen["chunks_top_k"] == 30
    assert seen["rerank"] == ("m", "q", 30)


@pytest.mark.asyncio
async def test_factory_precise_turns_on_bm25_and_rerank(monkeypatch):
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.search.types import SearchType

    monkeypatch.delenv("COGNEE_PRECISE_RETRIEVAL", raising=False)
    config = {"precise": True}

    hybrid = await mod.get_search_type_retriever_instance(
        SearchType.HYBRID_COMPLETION, query_text="q", retriever_specific_config=config
    )
    chunks = await mod.get_search_type_retriever_instance(
        SearchType.CHUNKS, query_text="q", retriever_specific_config=config
    )
    plain = await mod.get_search_type_retriever_instance(
        SearchType.HYBRID_COMPLETION, query_text="q"
    )

    assert hybrid.lexical_lane is True
    assert hybrid.rerank_model == PRECISE_RERANK_MODEL
    assert hybrid.rerank_top_n == PRECISE_RERANK_TOP_N
    assert chunks.lexical_fusion is True
    assert chunks.rerank_model == PRECISE_RERANK_MODEL
    assert plain.rerank_model is None
    assert plain.lexical_lane is False
