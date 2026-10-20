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

    async def boom():
        raise RuntimeError("graph unreadable")

    monkeypatch.setattr(hybrid_chunks, "_lexical_index", boom)
    assert await hybrid_chunks.search_lexical("q", 5) == []


class _Graph:
    __slots__ = ("counts", "fail", "key")  # like _GraphEngineHandle: no weak references

    def __init__(self, counts=(3, 2), fail=False):
        self.counts = counts
        self.fail = fail
        self.key = "db-1"

    async def get_graph_counts(self):
        if self.fail:
            raise RuntimeError("no counts")
        return self.counts


class _Index:
    built = 0

    def __init__(self, with_scores=False):
        _Index.built += 1

    async def initialize(self):
        pass

    async def get_retrieved_objects(self, query, *, top_k=None):
        return [({"id": "a"}, 1.0)][:top_k]


@pytest.fixture
def lexical_env(monkeypatch):
    import cognee.infrastructure.databases.graph as graph_pkg
    from cognee.modules.retrieval import bm25_retriever

    graph = _Graph()

    async def get_graph_engine():
        return graph

    _Index.built = 0
    monkeypatch.setattr(graph_pkg, "get_graph_engine", get_graph_engine)
    monkeypatch.setattr(bm25_retriever, "BM25ChunksRetriever", _Index)
    monkeypatch.setattr(hybrid_chunks, "_lexical_indexes", hybrid_chunks.OrderedDict())
    monkeypatch.setattr(hybrid_chunks, "_graph_key", lambda: graph.key)
    return graph


@pytest.mark.asyncio
async def test_lexical_index_is_reused_while_the_graph_is_unchanged(lexical_env):
    assert await hybrid_chunks.search_lexical("q", 5) == [({"id": "a"}, 1.0)]
    await hybrid_chunks.search_lexical("q", 5)

    assert _Index.built == 1


@pytest.mark.asyncio
async def test_lexical_index_is_rebuilt_when_the_graph_changes(lexical_env):
    await hybrid_chunks.search_lexical("q", 5)
    lexical_env.counts = (4, 2)
    await hybrid_chunks.search_lexical("q", 5)

    assert _Index.built == 2


@pytest.mark.asyncio
async def test_lexical_index_is_rebuilt_after_the_ttl(lexical_env, monkeypatch):
    await hybrid_chunks.search_lexical("q", 5)
    monkeypatch.setattr(hybrid_chunks, "LEXICAL_INDEX_TTL_SECONDS", -1.0)
    await hybrid_chunks.search_lexical("q", 5)

    assert _Index.built == 2


@pytest.mark.asyncio
async def test_lexical_index_is_not_cached_without_graph_counts(lexical_env):
    lexical_env.fail = True
    await hybrid_chunks.search_lexical("q", 5)
    await hybrid_chunks.search_lexical("q", 5)

    assert _Index.built == 2


@pytest.mark.asyncio
async def test_lexical_index_is_per_database(lexical_env):
    await hybrid_chunks.search_lexical("q", 5)
    lexical_env.key = "db-2"
    await hybrid_chunks.search_lexical("q", 5)
    lexical_env.key = "db-1"
    await hybrid_chunks.search_lexical("q", 5)

    assert _Index.built == 2


@pytest.mark.asyncio
async def test_lexical_index_cache_is_bounded(lexical_env, monkeypatch):
    monkeypatch.setattr(hybrid_chunks, "LEXICAL_INDEX_MAX_ENTRIES", 2)
    for key in ("a", "b", "c"):
        lexical_env.key = key
        await hybrid_chunks.search_lexical("q", 5)

    assert list(hybrid_chunks._lexical_indexes) == ["b", "c"]


def test_graph_key_follows_the_graph_context_config(monkeypatch):
    from cognee.infrastructure.databases.graph import config as graph_config

    monkeypatch.setattr(graph_config, "get_graph_context_config", lambda: {"path": "/a"})
    first = hybrid_chunks._graph_key()
    monkeypatch.setattr(graph_config, "get_graph_context_config", lambda: {"path": "/b"})

    assert hybrid_chunks._graph_key() != first


def test_lane_weights_scale_each_lanes_rrf_term():
    pairs = chunk_summary_pairs([_chunk("a"), _chunk("b")], [])
    add_lexical_ranks(pairs, [({"id": "b"}, 5.0)])

    lexical_off = rank_chunk_summary_pairs(
        pairs, limit=2, use_importance_weight=False, lane_weights=(1.0, 1.0, 0.0)
    )
    lexical_heavy = rank_chunk_summary_pairs(
        pairs, limit=2, use_importance_weight=False, lane_weights=(1.0, 1.0, 3.0)
    )

    # a: dense rank 0. b: dense rank 1 plus lexical rank 0, so b leads unless
    # the lexical term is weighted out.
    assert [pair["chunk_id"] for pair in lexical_off] == ["a", "b"]
    assert [pair["chunk_id"] for pair in lexical_heavy] == ["b", "a"]


def test_unit_lane_weights_match_the_default():
    pairs = chunk_summary_pairs([_chunk("a"), _chunk("b"), _chunk("c")], [])
    add_lexical_ranks(pairs, [({"id": "c"}, 2.0), ({"id": "a"}, 1.0)])

    default = rank_chunk_summary_pairs(pairs, limit=3, use_importance_weight=False)
    unit = rank_chunk_summary_pairs(
        pairs, limit=3, use_importance_weight=False, lane_weights=(1.0, 1.0, 1.0)
    )

    assert [p["chunk_id"] for p in default] == [p["chunk_id"] for p in unit]
