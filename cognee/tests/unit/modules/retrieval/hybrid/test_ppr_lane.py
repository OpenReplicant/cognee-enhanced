import pytest

from cognee.modules.retrieval.hybrid import ppr
from cognee.modules.retrieval.hybrid.pairs import (
    add_ppr_ranks,
    chunk_summary_pairs,
    source_chunk_ids_to_load,
)
from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs

NODES = [
    ("c1", {"type": "DocumentChunk"}),
    ("c2", {"type": "DocumentChunk"}),
    ("c3", {"type": "DocumentChunk"}),
    ("c4", {"type": "DocumentChunk"}),
    ("audrey", {"type": "Entity"}),
    ("dog", {"type": "Entity"}),
    ("vet", {"type": "Entity"}),
    ("person", {"type": "EntityType"}),
    ("doc", {"type": "TextDocument"}),
]
EDGES = [
    ("c1", "audrey", "contains", {}),
    ("c2", "dog", "contains", {}),
    ("audrey", "dog", "owns", {}),
    ("c3", "vet", "contains", {}),
    ("dog", "vet", "visited", {}),
    # Hub edges that must not carry mass: every chunk to its document, every
    # entity to its type.
    ("c1", "doc", "is_part_of", {}),
    ("c4", "doc", "is_part_of", {}),
    ("audrey", "person", "is_a", {}),
    ("vet", "person", "is_a", {}),
]


class _Graph:
    async def get_graph_data(self):
        return NODES, EDGES


@pytest.mark.asyncio
async def test_walk_graph_keeps_chunks_entities_and_drops_hub_types():
    walk = await ppr._build_walk_graph(_Graph())

    assert set(walk.graph.nodes) == {"c1", "c2", "c3", "c4", "audrey", "dog", "vet"}
    assert walk.chunk_ids == {"c1", "c2", "c3", "c4"}
    assert not walk.graph.has_edge("c1", "c4")


@pytest.mark.asyncio
async def test_ppr_ranks_chunks_by_graph_distance_from_the_seeds():
    walk = await ppr._build_walk_graph(_Graph())

    ranked = ppr.rank_chunks(walk, {"audrey": 1.0}, limit=10)

    # c1 holds the seed, c2 is one relation away, c3 two; c4 is unreachable.
    assert ranked == ["c1", "c2", "c3"]


@pytest.mark.asyncio
async def test_ppr_ignores_unknown_seeds_and_zero_limits():
    walk = await ppr._build_walk_graph(_Graph())

    assert ppr.rank_chunks(walk, {"nobody": 1.0}, limit=10) == []
    assert ppr.rank_chunks(walk, {"audrey": 1.0}, limit=0) == []


@pytest.mark.asyncio
async def test_search_ppr_fails_open(monkeypatch):
    async def boom():
        raise RuntimeError("graph unreadable")

    monkeypatch.setattr(ppr, "walk_graph", boom)

    assert await ppr.search_ppr([{"id": "audrey"}], 5) == []
    assert await ppr.search_ppr([], 5) == []


def test_ppr_ranks_join_the_rrf_and_ppr_only_chunks_are_loaded():
    pairs = chunk_summary_pairs([{"id": "a", "text": "a"}, {"id": "b", "text": "b"}], [])
    add_ppr_ranks(pairs, ["b", "z"])

    ranked = rank_chunk_summary_pairs(pairs, limit=3, use_importance_weight=False)

    assert [pair["chunk_id"] for pair in ranked][:1] == ["b"]
    assert source_chunk_ids_to_load(pairs) == ["z"]


def test_three_lane_weights_still_accepted_with_the_ppr_lane():
    pairs = chunk_summary_pairs([{"id": "a", "text": "a"}], [])
    add_ppr_ranks(pairs, ["a"])

    ranked = rank_chunk_summary_pairs(
        pairs, limit=1, use_importance_weight=False, lane_weights=(1.0, 1.0, 1.0)
    )

    assert [pair["chunk_id"] for pair in ranked] == ["a"]


def test_personalized_pagerank_matches_networkx_when_scipy_is_present():
    networkx = pytest.importorskip("networkx")
    pytest.importorskip("scipy")

    graph = networkx.karate_club_graph()
    graph = networkx.relabel_nodes(graph, str)
    graph.add_node("isolated")
    seeds = {"0": 1.0, "33": 0.5, "isolated": 0.25}

    ours = ppr.personalized_pagerank(graph, seeds, alpha=0.5, tol=1e-10)
    theirs = networkx.pagerank(graph, alpha=0.5, personalization=seeds, tol=1e-10, weight=None)

    assert all(abs(ours[node] - theirs[node]) < 1e-6 for node in graph)


def test_personalized_pagerank_needs_no_scipy(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.split(".")[0] == "scipy":
            raise ModuleNotFoundError("No module named 'scipy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    import networkx

    graph = networkx.path_graph(["a", "b", "c"])
    scores = ppr.personalized_pagerank(graph, {"a": 1.0})

    assert scores["a"] > scores["b"] > scores["c"] > 0
