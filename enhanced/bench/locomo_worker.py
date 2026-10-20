"""One LoCoMo conversation, in one process: ingest it, or retrieve for its questions.

Run through ``enhanced.bench.run``, which sets the profile env (data dirs,
extractor, embeddings) before this module imports cognee.

    python -m enhanced.bench.locomo_worker ingest   --conv 0 --store DIR [--max-sessions N]
    python -m enhanced.bench.locomo_worker retrieve --conv 0 --store DIR --variant hybrid --out F

Ingestion mirrors ``cognee/eval_framework/locomo/ingest.py`` (overview + one
document per 6-turn window, chunk_size 4096 so each window is one chunk) but
skips ``improve()``: retrieval benchmarks measure the cognify output alone.

Retrieval writes one JSON line per question: the gold windows (from the
question's evidence dia_ids) and the ranked windows the variant returned.
"""

import argparse
import asyncio
import importlib
import json
import re
import time
from pathlib import Path

from cognee.eval_framework.benchmark_adapters.locomo_adapter import LocomoAdapter
from cognee.eval_framework.locomo.preprocess import (
    build_conversation_windows,
    conversation_overview_text,
)
from enhanced.bench.profiles import CACHE_DIR
from enhanced.bench.variants import VARIANTS

WINDOW_TURNS = 6
CHUNK_SIZE = 4096
OVERVIEW = -1  # window index of the overview document (holds no evidence)


def _adapter(max_sessions):
    data_path = CACHE_DIR / "datasets" / "locomo10.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    return LocomoAdapter(max_sessions=max_sessions, data_path=str(data_path))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _dia(dia_id: str):
    match = re.fullmatch(r"D(\d+):(\d+)", dia_id.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def gold_windows(evidence: list[str], window_map: list[dict]) -> list[int]:
    gold = set()
    for dia_id in evidence:
        parsed = _dia(dia_id)
        if parsed is None:
            continue
        session, turn = parsed
        for index, window in enumerate(window_map):
            first, last = _dia(window["first_dia_id"]), _dia(window["last_dia_id"])
            if window["session"] == session and first[1] <= turn <= last[1]:
                gold.add(index)
    return sorted(gold)


def _text_of(item) -> str:
    if isinstance(item, tuple):
        return _text_of(item[0])
    payload = getattr(item, "payload", None)
    if payload is None and isinstance(item, dict):
        payload = item.get("payload", item)
    return (payload or {}).get("text", "") if isinstance(payload, dict) else ""


def _ranked_chunks(retrieved) -> list:
    if isinstance(retrieved, dict):  # hybrid-family: {"chunks": [...], ...}
        return list(retrieved.get("chunks", []))
    return list(retrieved or [])


async def ingest(args) -> None:
    import cognee

    adapter = _adapter(args.max_sessions)
    conversation = adapter.load_conversation(args.conv)
    windows = build_conversation_windows(conversation, WINDOW_TURNS)
    texts = [conversation_overview_text(conversation)] + [w.text for w in windows]

    started = time.monotonic()
    await cognee.remember(
        texts,
        dataset_name=f"locomo_conv{args.conv}",
        chunk_size=CHUNK_SIZE,
        self_improvement=False,
    )
    meta = {
        "conv": args.conv,
        "sample_id": conversation.sample_id,
        "max_sessions": args.max_sessions,
        "window_turns": WINDOW_TURNS,
        "chunk_size": CHUNK_SIZE,
        "ingest_seconds": round(time.monotonic() - started, 1),
        "overview_text": conversation_overview_text(conversation),
        "windows": [
            {
                "session": w.session_index,
                "first_dia_id": w.first_dia_id,
                "last_dia_id": w.last_dia_id,
                "text": w.text,
            }
            for w in windows
        ],
    }
    Path(args.store, "ingest_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"ingested conv {args.conv}: {len(windows)} windows in {meta['ingest_seconds']}s")


async def retrieve(args) -> None:
    meta = json.loads(Path(args.store, "ingest_meta.json").read_text())
    window_map = meta["windows"]
    by_text = {_norm(w["text"]): i for i, w in enumerate(window_map)}
    by_text[_norm(meta["overview_text"])] = OVERVIEW

    def window_of(text: str):
        key = _norm(text)
        if key in by_text:
            return by_text[key]
        for known, index in by_text.items():  # chunker may trim the edges
            if key and (key in known or known in key):
                return index
        return None

    spec = VARIANTS[args.variant]
    module_name, class_name = spec["retriever"].split(":")
    retriever = getattr(importlib.import_module(module_name), class_name)(**spec["kwargs"])

    adapter = _adapter(meta["max_sessions"])
    conversation = adapter.load_conversation(args.conv)
    questions = adapter.questions_for(conversation)

    unmapped = 0
    with open(args.out, "w") as out:
        for q in questions:
            started = time.monotonic()
            retrieved = await retriever.get_retrieved_objects(q["question"])
            latency = time.monotonic() - started
            ranked = []
            for chunk in _ranked_chunks(retrieved):
                index = window_of(_text_of(chunk))
                if index is None:
                    unmapped += 1
                elif index not in ranked:
                    ranked.append(index)
            out.write(
                json.dumps(
                    {
                        "qid": f"conv{args.conv}:{q['question_idx']}",
                        "category": q["question_type"],
                        "gold": gold_windows(q["evidence"], window_map),
                        "ranked": ranked,
                        "latency_s": round(latency, 4),
                    }
                )
                + "\n"
            )
    if unmapped:
        print(f"WARNING: {unmapped} retrieved chunks did not map to a window")
    print(f"retrieved conv {args.conv} variant {args.variant}: {len(questions)} questions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest", "retrieve"])
    parser.add_argument("--conv", type=int, required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--variant", default="hybrid")
    parser.add_argument("--out")
    args = parser.parse_args()
    asyncio.run(ingest(args) if args.command == "ingest" else retrieve(args))


if __name__ == "__main__":
    main()
