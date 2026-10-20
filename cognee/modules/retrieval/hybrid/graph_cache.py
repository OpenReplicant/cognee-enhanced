"""Per-database caches for structures the hybrid lanes build from the whole graph.

The lexical lane's BM25 index and the PPR lane's graph are built from every
node of the current graph database. Each is kept once per database, reused
while the graph's node and edge counts are unchanged and the entry is younger
than the TTL (the TTL covers an edit that leaves both counts equal), in a small
LRU per cache. The key is a hash of the graph context config, which is what
identifies the database: ``get_graph_engine`` returns a fresh handle per call.
"""

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")

TTL_SECONDS = 300.0
MAX_ENTRIES = 16
_caches: dict[str, "OrderedDict[str, tuple]"] = {}


def graph_key() -> str:
    from cognee.infrastructure.databases.graph.config import get_graph_context_config

    config = json.dumps(get_graph_context_config(), sort_keys=True, default=str)
    return hashlib.sha256(config.encode()).hexdigest()


async def cached_for_graph(name: str, build: Callable[[Any], Awaitable[Any]]) -> Any:
    """The cached ``name`` structure for the current graph, built by ``build(engine)``."""
    from cognee.infrastructure.databases.graph import get_graph_engine

    key = graph_key()
    graph_engine = await get_graph_engine()
    try:
        counts = await graph_engine.get_graph_counts()
    except Exception:
        # Cannot tell whether the graph changed, so never reuse a cached build.
        logger.info("Graph counts unavailable; %s not cached", name, exc_info=True)
        counts = None
    cache = _caches.setdefault(name, OrderedDict())
    entry = cache.get(key)
    now = time.monotonic()
    if (
        entry is not None
        and counts is not None
        and entry[0] == counts
        and now - entry[1] < TTL_SECONDS
    ):
        cache.move_to_end(key)
        return entry[2]
    value = await build(graph_engine)
    if counts is not None:
        cache[key] = (counts, now, value)
        cache.move_to_end(key)
        while len(cache) > MAX_ENTRIES:
            cache.popitem(last=False)
    return value


def clear() -> None:
    _caches.clear()
