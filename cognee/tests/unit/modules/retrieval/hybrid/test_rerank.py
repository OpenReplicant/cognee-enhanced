import pytest

from cognee.modules.retrieval.hybrid import rerank


def _chunk(chunk_id: str) -> dict:
    return {"id": chunk_id, "text": f"text {chunk_id}"}


@pytest.mark.asyncio
async def test_rerank_orders_by_score_and_keeps_fused_order_on_ties(monkeypatch):
    scores = {"text a": 0.1, "text b": 0.9, "text c": 0.1}
    monkeypatch.setattr(rerank, "_score", lambda model, query, texts: [scores[t] for t in texts])

    ranked = await rerank.rerank_chunks("m", "q", [_chunk("a"), _chunk("b"), _chunk("c")])

    assert [chunk["id"] for chunk in ranked] == ["b", "a", "c"]


@pytest.mark.asyncio
async def test_rerank_failure_keeps_fused_order(monkeypatch):
    def boom(model, query, texts):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(rerank, "_score", boom)
    chunks = [_chunk("a"), _chunk("b")]

    assert await rerank.rerank_chunks("m", "q", chunks) == chunks


@pytest.mark.asyncio
async def test_rerank_skips_the_model_for_fewer_than_two_chunks(monkeypatch):
    def unexpected(model, query, texts):
        raise AssertionError("scored a single chunk")

    monkeypatch.setattr(rerank, "_score", unexpected)
    assert await rerank.rerank_chunks("m", "q", [_chunk("a")]) == [_chunk("a")]


@pytest.mark.asyncio
async def test_rerank_top_n_reorders_only_the_head(monkeypatch):
    seen = []

    def score(model, query, texts):
        seen.append(texts)
        return [0.1, 0.9]

    monkeypatch.setattr(rerank, "_score", score)
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]

    ranked = await rerank.rerank_chunks("m", "q", chunks, top_n=2, max_chars=4)

    assert [chunk["id"] for chunk in ranked] == ["b", "a", "c"]
    assert seen == [["text", "text"]]
