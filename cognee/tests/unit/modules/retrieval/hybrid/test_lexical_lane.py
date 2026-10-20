import pytest

from cognee.modules.retrieval.hybrid import chunks as hybrid_chunks
from cognee.modules.retrieval.hybrid.pairs import (
    add_lexical_ranks,
    chunk_summary_pairs,
    source_chunk_ids_to_load,
)
from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs


def _chunk(chunk_id: str) -> dict:
    return {"id": chunk_id, "text": f"text {chunk_id}"}


def test_lexical_ranks_skip_zero_scores_and_rank_the_rest_densely():
    pairs = chunk_summary_pairs([_chunk("a")], [])
    add_lexical_ranks(pairs, [({"id": "z"}, 0.0), ({"id": "b"}, 3.0), ({"id": "a"}, 1.0)])

    by_id = {pair["chunk_id"]: pair for pair in pairs}
    assert "z" not in by_id
    assert by_id["b"]["lexical_rank"] == 0
    assert by_id["a"]["lexical_rank"] == 1
    assert by_id["a"]["vector_rank"] == 0


def test_lexical_only_chunks_are_loaded_from_the_vector_store():
    pairs = chunk_summary_pairs([_chunk("a")], [])
    add_lexical_ranks(pairs, [({"id": "b"}, 2.0)])

    assert source_chunk_ids_to_load(pairs) == ["b"]


def test_lexical_ranks_respect_the_node_filter():
    pairs = chunk_summary_pairs([], [])
    hits = [({"id": "in", "belongs_to_set": ["p1"]}, 2.0), ({"id": "out"}, 3.0)]
    add_lexical_ranks(pairs, hits, node_name=["p1"])

    assert [pair["chunk_id"] for pair in pairs] == ["in"]


def test_lexical_agreement_lifts_a_chunk_in_rrf():
    pairs = chunk_summary_pairs([_chunk("a"), _chunk("b")], [])
    add_lexical_ranks(pairs, [({"id": "b"}, 5.0)])

    ranked = rank_chunk_summary_pairs(pairs, limit=2, use_importance_weight=False)

    assert [pair["chunk_id"] for pair in ranked] == ["b", "a"]


def test_pairs_without_a_lexical_rank_key_still_rank():
    legacy = {"chunk": _chunk("a"), "chunk_id": "a", "vector_rank": 0, "summary_rank": None}

    ranked = rank_chunk_summary_pairs([legacy], limit=1, use_importance_weight=False)

    assert [pair["chunk_id"] for pair in ranked] == ["a"]


@pytest.mark.asyncio
async def test_search_lexical_is_off_at_zero_limit_and_fails_open(monkeypatch):
    assert await hybrid_chunks.search_lexical("q", 0) == []

    async def boom(self, query):
        raise RuntimeError("graph unreadable")

    from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever

    monkeypatch.setattr(BM25ChunksRetriever, "get_retrieved_objects", boom)
    assert await hybrid_chunks.search_lexical("q", 5) == []
