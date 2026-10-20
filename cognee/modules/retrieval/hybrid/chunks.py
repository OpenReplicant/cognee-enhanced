import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Iterable
from itertools import chain
from typing import Any

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.chunking.models.DltRow import DltRow
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.retrieval.hybrid.pairs import (
    add_lexical_ranks,
    attach_source_chunks,
    chunk_summary_pairs,
    source_chunk_ids_to_load,
    summary_id_for_chunk,
    summary_text_by_chunk_id,
)
from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs
from cognee.modules.retrieval.hybrid.results import payload_matches_node_filter
from cognee.modules.retrieval.utils.results import display_value, payload, result_id
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")


def chunk_collections(include_dlt_rows: bool) -> tuple[str, ...]:
    """The chunk lane's collections: document chunks, plus DLT rows when the dataset has any."""
    collections = (DocumentChunk.vector_collection(),)
    return collections + (DltRow.vector_collection(),) if include_dlt_rows else collections


async def dlt_rows_indexed(vector_engine: Any) -> bool:
    """Whether the dataset holds DLT rows: their collection exists. Checked once per search."""
    return await vector_engine.has_collection(DltRow.vector_collection())


async def retrieve_hybrid_chunks(
    vector_engine: Any,
    query: str,
    chunks_top_k: int,
    text_summaries_top_k: int | None,
    node_name: list[str] | None,
    node_name_filter_operator: str,
    use_importance_weight: bool,
    query_vector: list[float] | None = None,
    use_truth_weight: bool = False,
    q_coords: list[float] | None = None,
    truth_state_by_id: dict | None = None,
    current_truth_epoch: int | None = None,
    personal_weights: dict | None = None,
    personal_influence: float = 0.0,
    collections: tuple[str, ...] = chunk_collections(include_dlt_rows=False),
    lexical_lane: bool = False,
    lane_weights: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    candidate_limit = chunk_candidate_limit(chunks_top_k)
    summary_limit = summary_candidate_limit(chunks_top_k, text_summaries_top_k)
    lexical_limit = candidate_limit if lexical_lane else 0
    vector_chunks, summary_hits, lexical_hits = await asyncio.gather(
        search_collections(
            vector_engine,
            collections,
            query,
            candidate_limit,
            node_name,
            node_name_filter_operator,
            query_vector=query_vector,
        ),
        search_collection(
            vector_engine,
            "TextSummary_text",
            query,
            summary_limit,
            node_name,
            node_name_filter_operator,
            query_vector=query_vector,
        ),
        search_lexical(query, lexical_limit),
    )

    pairs = chunk_summary_pairs(
        vector_chunks,
        summary_hits,
        node_name,
        node_name_filter_operator,
    )
    add_lexical_ranks(pairs, lexical_hits, node_name, node_name_filter_operator)
    missing_source_chunk_ids = source_chunk_ids_to_load(pairs)
    if missing_source_chunk_ids:
        source_chunks = await load_source_chunks_for_summaries(
            vector_engine,
            missing_source_chunk_ids,
            node_name,
            node_name_filter_operator,
        )
        attach_source_chunks(pairs, source_chunks)

    ranked_pairs = rank_chunk_summary_pairs(
        pairs,
        chunks_top_k,
        use_importance_weight,
        use_truth_weight=use_truth_weight,
        q_coords=q_coords,
        truth_state_by_id=truth_state_by_id,
        current_truth_epoch=current_truth_epoch,
        personal_weights=personal_weights,
        personal_influence=personal_influence,
        lane_weights=lane_weights,
    )
    if summary_limit > 0:
        await load_summary_text_for_ranked_pairs(
            vector_engine,
            ranked_pairs,
            node_name,
            node_name_filter_operator,
        )

    return {
        "chunks": [pair["chunk"] for pair in ranked_pairs if pair["chunk"] is not None],
        "chunk_summaries": summary_text_by_chunk_id(ranked_pairs),
    }


# One BM25 index per graph database, reused while the graph's node and edge
# counts are unchanged and the entry is younger than the TTL (the TTL covers an
# edit that leaves both counts equal). Keyed by a hash of the graph context
# config, which is what identifies the database (``get_graph_engine`` returns
# a fresh handle per call); a small LRU bounds the memory.
LEXICAL_INDEX_TTL_SECONDS = 300.0
LEXICAL_INDEX_MAX_ENTRIES = 16
_lexical_indexes: "OrderedDict[str, tuple]" = OrderedDict()


def _graph_key() -> str:
    from cognee.infrastructure.databases.graph.config import get_graph_context_config

    config = json.dumps(get_graph_context_config(), sort_keys=True, default=str)
    return hashlib.sha256(config.encode()).hexdigest()


async def _lexical_index():
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever

    key = _graph_key()
    graph_engine = await get_graph_engine()
    try:
        counts = await graph_engine.get_graph_counts()
    except Exception:
        # Cannot tell whether the graph changed, so never reuse an index.
        logger.info("Graph counts unavailable; lexical index not cached", exc_info=True)
        counts = None
    entry = _lexical_indexes.get(key)
    now = time.monotonic()
    if (
        entry is not None
        and counts is not None
        and entry[0] == counts
        and now - entry[1] < LEXICAL_INDEX_TTL_SECONDS
    ):
        _lexical_indexes.move_to_end(key)
        return entry[2]
    index = BM25ChunksRetriever(with_scores=True)
    await index.initialize()
    if counts is not None:
        _lexical_indexes[key] = (counts, now, index)
        _lexical_indexes.move_to_end(key)
        while len(_lexical_indexes) > LEXICAL_INDEX_MAX_ENTRIES:
            _lexical_indexes.popitem(last=False)
    return index


async def search_lexical(query: str, limit: int) -> list[tuple[dict, float]]:
    """BM25 over every DocumentChunk, as ``(payload, score)`` pairs; empty when off.

    A lexical failure (no chunks yet, graph unreadable) leaves the dense lanes
    to answer alone rather than failing the search.
    """
    if limit <= 0:
        return []
    try:
        index = await _lexical_index()
        return await index.get_retrieved_objects(query, top_k=limit)
    except Exception:
        logger.warning("Hybrid lexical lane failed; continuing without it", exc_info=True)
        return []


def chunk_candidate_limit(chunks_top_k: int) -> int:
    return max(0, chunks_top_k * 2)


def summary_candidate_limit(chunks_top_k: int, text_summaries_top_k: int | None) -> int:
    if text_summaries_top_k is None:
        return max(0, chunks_top_k)
    return text_summaries_top_k


def _score(result: Any) -> float:
    score = getattr(result, "score", None)
    return float(score) if isinstance(score, (int, float)) else float("inf")


def merge_scored(results: Iterable[Any], limit: int) -> list[Any]:
    """One ranked list from hits of several collections in the same embedding space:
    lower score first (a stable sort, so hits without a numeric score keep their
    channel order at the end), one hit per node, cut to ``limit``."""
    merged = []
    seen = set()
    for result in sorted(results, key=_score):
        key = result_id(result) or id(result)
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= limit:
            break
    return merged


async def search_collections(
    vector_engine: Any,
    collection_names: Iterable[str],
    query: str,
    limit: int,
    node_name: list[str] | None,
    node_name_filter_operator: str,
    *,
    apply_node_filter: bool = True,
    query_vector: list[float] | None = None,
) -> list[Any]:
    """``search_collection`` over several collections, merged into one ranked list."""
    hits = await asyncio.gather(
        *(
            search_collection(
                vector_engine,
                collection_name,
                query,
                limit,
                node_name,
                node_name_filter_operator,
                apply_node_filter=apply_node_filter,
                query_vector=query_vector,
            )
            for collection_name in collection_names
        )
    )
    return merge_scored(chain.from_iterable(hits), limit)


async def search_collection(
    vector_engine: Any,
    collection_name: str,
    query: str,
    limit: int,
    node_name: list[str] | None,
    node_name_filter_operator: str,
    *,
    apply_node_filter: bool = True,
    query_vector: list[float] | None = None,
) -> list[Any]:
    if limit <= 0:
        return []

    search_node_name = node_name if apply_node_filter else None
    search_operator = node_name_filter_operator if apply_node_filter else "OR"
    try:
        return await vector_engine.search(
            collection_name,
            None if query_vector is not None else query,
            query_vector=query_vector,
            limit=limit,
            include_payload=True,
            node_name=search_node_name,
            node_name_filter_operator=search_operator,
        )
    except CollectionNotFoundError:
        logger.debug("%s collection not found; using empty channel", collection_name)
        return []


async def load_source_chunks_for_summaries(
    vector_engine: Any,
    chunk_ids: list[str],
    node_name: list[str] | None,
    node_name_filter_operator: str,
) -> list[Any]:
    chunks = await vector_engine.retrieve("DocumentChunk_text", chunk_ids)
    found_ids = {result_id(chunk) for chunk in chunks}
    missing_ids = sorted(set(chunk_ids) - {chunk_id for chunk_id in found_ids if chunk_id})
    if missing_ids:
        logger.warning(
            "TextSummary_text hit referenced missing DocumentChunk_text row(s): %s",
            missing_ids,
        )

    source_chunks = []
    filtered_ids = []
    for chunk in chunks:
        if payload_matches_node_filter(payload(chunk), node_name, node_name_filter_operator):
            source_chunks.append(chunk)
            continue

        chunk_id = result_id(chunk)
        if chunk_id:
            filtered_ids.append(chunk_id)

    if filtered_ids:
        logger.warning(
            "TextSummary_text source chunk failed node filter: %s",
            sorted(filtered_ids),
        )
    return source_chunks


async def load_summary_text_for_ranked_pairs(
    vector_engine: Any,
    ranked_pairs: list[dict],
    node_name: list[str] | None,
    node_name_filter_operator: str,
) -> None:
    summary_ids_by_chunk_id = {}
    for pair in ranked_pairs:
        if pair["summary_text"]:
            continue

        chunk_id = pair["chunk_id"]
        if not chunk_id:
            continue

        summary_id = pair["summary_id"] or summary_id_for_chunk(chunk_id)
        if summary_id is None:
            logger.debug("Cannot fetch paired TextSummary for non-UUID chunk id %s", chunk_id)
            continue

        pair["summary_id"] = summary_id
        summary_ids_by_chunk_id[chunk_id] = summary_id

    if not summary_ids_by_chunk_id:
        return

    try:
        summaries = await vector_engine.retrieve(
            "TextSummary_text",
            list(summary_ids_by_chunk_id.values()),
        )
    except CollectionNotFoundError:
        logger.warning("TextSummary_text collection missing while loading chunk summaries")
        return

    summaries_by_id = {result_id(summary): summary for summary in summaries}
    for pair in ranked_pairs:
        chunk_id = pair["chunk_id"]
        summary_id = pair["summary_id"]
        if not chunk_id or not summary_id:
            continue

        summary = summaries_by_id.get(summary_id)
        if summary is None:
            # A summary is optional enrichment: DLT rows and custom pipelines write none.
            continue

        summary_payload = payload(summary)
        summary_text = display_value(summary_payload.get("text"))
        if not summary_text:
            logger.warning(
                "Paired TextSummary_text row has no text: chunk_id=%s summary_id=%s",
                chunk_id,
                summary_id,
            )
            continue

        if not payload_matches_node_filter(summary_payload, node_name, node_name_filter_operator):
            logger.warning(
                "Paired TextSummary_text row failed node filter: chunk_id=%s summary_id=%s",
                chunk_id,
                summary_id,
            )
            continue

        pair["summary_text"] = summary_text
