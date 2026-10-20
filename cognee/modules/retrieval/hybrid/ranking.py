from cognee.modules.retrieval.utils.results import payload, result_id
from cognee.modules.truth_subspace.align import truth_factor
from cognee.modules.user_preferences import personal_factor


def rank_chunk_summary_pairs(
    pairs: list[dict],
    limit: int,
    use_importance_weight: bool,
    use_truth_weight: bool = False,
    q_coords: list[float] | None = None,
    truth_state_by_id: dict | None = None,
    current_truth_epoch: int | None = None,
    personal_weights: dict | None = None,
    personal_influence: float = 0.0,
    lane_weights: tuple[float, float, float] | None = None,
) -> list[dict]:
    """Rank chunk-lane pairs by reciprocal-rank fusion of their lane ranks.

    ``lane_weights`` scales each lane's RRF term as (dense chunk, dense
    summary, lexical); None weighs all three 1.0 (enhanced/HYPOTHESES.md H2).
    """
    if limit <= 0:
        return []
    # The optional PPR lane (H6) always weighs 1.0.
    weights = tuple(lane_weights or (1.0, 1.0, 1.0)) + (1.0,)

    rrf_k = _rrf_k(limit)
    ranked = []
    for pair in pairs:
        chunk = pair["chunk"]
        if chunk is None:
            continue

        lane_ranks = (
            pair["vector_rank"],
            pair["summary_rank"],
            pair.get("lexical_rank"),
            pair.get("ppr_rank"),
        )
        weighted = [(weight, rank) for weight, rank in zip(weights, lane_ranks) if rank is not None]
        ranks = [rank for _, rank in weighted]
        if not ranks:
            continue

        rrf_score = sum(weight / (rrf_k + rank + 1) for weight, rank in weighted)
        final_score = rrf_score
        if use_importance_weight:
            final_score *= _importance_factor(chunk)

        chunk_id = pair["chunk_id"] or result_id(chunk) or ""
        if use_truth_weight and q_coords and current_truth_epoch is not None:
            truth_state = (truth_state_by_id or {}).get(chunk_id, {})
            if truth_state.get("truth_epoch") == current_truth_epoch:
                final_score *= truth_factor(truth_state.get("truth_alignment", []), q_coords)

        if personal_weights:
            # Personal prefers weight, multiplied beside the importance and
            # truth factors. Hybrid sorts by descending score, hence score
            # space. Chunks with no matching weight keep their score, so an
            # empty map is byte-identical to an un-personalized run.
            personal_weight = personal_weights.get(chunk_id)
            if personal_weight is not None:
                final_score *= personal_factor(
                    personal_weight, personal_influence, distance_space=False
                )

        ranked.append((final_score, rrf_score, min(ranks), chunk_id, pair))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return [pair for *_, pair in ranked[:limit]]


def _rrf_k(chunks_top_k: int) -> int:
    return max(30, min(60, 20 + 2 * chunks_top_k))


def _importance_factor(chunk) -> float:
    raw_importance = payload(chunk).get("importance_weight")
    importance = raw_importance if isinstance(raw_importance, (int, float)) else 0.5
    importance = max(0.0, min(1.0, importance))
    return 0.75 + 0.5 * importance
