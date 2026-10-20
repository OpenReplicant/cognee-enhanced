import random

from cognee.modules.retrieval.hybrid.entities import _edge_bullets_from_connections
from cognee.modules.retrieval.hybrid.facts import connection_edge_type_id


def _connection(source: str, relationship: str, target: str, text: str | None = None):
    edge = {"relationship_name": relationship}
    if text:
        edge["edge_text"] = text
    return ({"id": source, "name": source}, edge, {"id": target, "name": target})


CONNECTIONS = (
    [_connection("audrey", "owns", f"dog{i}") for i in range(8)]
    + [_connection("audrey", "participated_in", f"session{i}") for i in range(8)]
    + [
        _connection("audrey", "owns", "dog1", text="Audrey adopted dog1"),
    ]
)


def _texts(connections, max_edges):
    return [bullet["text"] for bullet in _edge_bullets_from_connections(connections, max_edges, {})]


def test_edge_bullets_do_not_depend_on_graph_return_order():
    expected = _texts(CONNECTIONS, 10)
    for seed in range(20):
        shuffled = list(CONNECTIONS)
        random.Random(seed).shuffle(shuffled)
        assert _texts(shuffled, 10) == expected


def test_duplicate_connection_keeps_the_same_text_whatever_the_order():
    forward = _texts(CONNECTIONS, 20)
    backward = _texts(list(reversed(CONNECTIONS)), 20)
    assert forward == backward
    assert sum("dog1" in text for text in forward) == 1


def test_query_ranked_edges_still_lead():
    # "zz_rel" sorts last by content, so it can only lead through its rank.
    special = _connection("audrey", "zz_rel", "zebra")
    ranked = {connection_edge_type_id(special[1]): 0}
    bullets = _edge_bullets_from_connections(list(CONNECTIONS) + [special], 3, ranked)
    assert "zebra" in bullets[0]["text"]
    unranked = _edge_bullets_from_connections(list(CONNECTIONS) + [special], 3, {})
    assert all("zebra" not in bullet["text"] for bullet in unranked)
