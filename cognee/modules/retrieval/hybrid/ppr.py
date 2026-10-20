"""Optional Personalized PageRank lane for the hybrid chunk RRF (enhanced/HYPOTHESES.md H6).

HippoRAG-style: the entities the query matches seed a random walk with restart
over the entity–chunk graph, and chunks are ranked by the stationary mass they
collect, so a passage can be found through the entities it shares with other
relevant passages even when it shares no words with the query.

The walk graph holds DocumentChunk, Entity and Timestamp nodes and every edge
among them, undirected. EntityType, TextDocument and TextSummary nodes are
left out: type nodes are hubs ("Person") that would spread mass to every
member. The graph is built once per database (``graph_cache``).
"""

from typing import Any

from cognee.modules.retrieval.hybrid import graph_cache
from cognee.modules.retrieval.utils.results import result_id
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")

WALK_NODE_TYPES = frozenset({"DocumentChunk", "Entity", "Timestamp"})
DAMPING = 0.5
SEED_ENTITIES = 10


class WalkGraph:
    """Undirected walk graph plus which of its nodes are chunks."""

    def __init__(self, graph, chunk_ids: frozenset[str]):
        self.graph = graph
        self.chunk_ids = chunk_ids


async def _build_walk_graph(graph_engine: Any) -> WalkGraph:
    import networkx as nx

    nodes, edges = await graph_engine.get_graph_data()
    node_type = {str(node_id): (data or {}).get("type") for node_id, data in nodes}
    keep = {node_id for node_id, kind in node_type.items() if kind in WALK_NODE_TYPES}
    graph = nx.Graph()
    graph.add_nodes_from(keep)
    for source, target, *_ in edges:
        source, target = str(source), str(target)
        if source in keep and target in keep and source != target:
            graph.add_edge(source, target)
    chunks = frozenset(n for n in keep if node_type[n] == "DocumentChunk")
    return WalkGraph(graph, chunks)


async def walk_graph() -> WalkGraph:
    return await graph_cache.cached_for_graph("ppr", _build_walk_graph)


def rank_chunks(walk: WalkGraph, seeds: dict[str, float], limit: int) -> list[str]:
    """Chunk ids by PPR mass from ``seeds`` (node id -> weight), best first."""
    import networkx as nx

    personalization = {node: weight for node, weight in seeds.items() if node in walk.graph}
    if not personalization or limit <= 0:
        return []
    scores = nx.pagerank(walk.graph, alpha=DAMPING, personalization=personalization)
    ranked = sorted(
        (node for node in walk.chunk_ids if scores.get(node, 0.0) > 0.0),
        key=lambda node: (-scores[node], node),
    )
    return ranked[:limit]


async def search_ppr(entity_hits: list[Any], limit: int) -> list[str]:
    """PPR-ranked chunk ids seeded by ``entity_hits`` (in rank order); empty when off.

    Any failure leaves the other lanes to answer alone.
    """
    if limit <= 0 or not entity_hits:
        return []
    seeds: dict[str, float] = {}
    for rank, hit in enumerate(entity_hits[:SEED_ENTITIES]):
        node_id = result_id(hit)
        if node_id and node_id not in seeds:
            seeds[node_id] = 1.0 / (rank + 1)
    try:
        return rank_chunks(await walk_graph(), seeds, limit)
    except Exception:
        logger.warning("Hybrid PPR lane failed; continuing without it", exc_info=True)
        return []
