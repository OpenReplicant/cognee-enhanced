"""One LongMemEval-S question, in one process: ingest its haystack, or retrieve.

    python -m enhanced.bench.longmemeval_worker ingest   --qid ID --store DIR
    python -m enhanced.bench.longmemeval_worker retrieve --qid ID --store DIR \\
        --variants hybrid,hybrid_bm25 --out F

Every LongMemEval question has its own haystack (~48 sessions, ~120k tokens),
so each question gets its own store. Ingestion is **chunks only**: cognify's
default tasks without graph extraction and summarisation (classify, chunk,
embed + store), the way cognee's DLT pipeline stores rows. No LLM is called,
so only the chunk lanes are measured: hybrid's entity and summary lanes are
empty here and ``hybrid`` reduces to dense chunks.

Documents are rounds (one user turn plus the assistant reply), each with a
dated header, following LongMemEval's finding that round-level keys with
timestamps retrieve better than whole sessions. Gold is session-level:
``answer_session_ids``. Retrieval writes one JSON line per variant.
"""

import argparse
import asyncio
import importlib
import json
import re
import time
from pathlib import Path

from enhanced.bench.profiles import CACHE_DIR
from enhanced.bench.variants import VARIANTS

DATA_PATH = CACHE_DIR / "datasets" / "longmemeval" / "longmemeval_s_cleaned.json"
CHUNK_SIZE = 4096


def load_questions() -> dict[str, dict]:
    return {q["question_id"]: q for q in json.loads(DATA_PATH.read_text())}


def category(question: dict) -> str:
    if question["question_id"].endswith("_abs"):
        return "abstention"
    return question["question_type"]


def rounds(question: dict) -> list[dict]:
    """Each user turn with the assistant reply that follows it, dated."""
    out = []
    for session_id, date, session in zip(
        question["haystack_session_ids"], question["haystack_dates"], question["haystack_sessions"]
    ):
        number = 0
        for i, turn in enumerate(session):
            if turn["role"] != "user":
                continue
            number += 1
            reply = session[i + 1]["content"] if i + 1 < len(session) else ""
            text = (
                f"Chat session {session_id}, {date}, round {number}.\n"
                f"User: {turn['content']}\nAssistant: {reply}"
            )
            out.append({"session_id": session_id, "date": date, "text": text})
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _text_of(item) -> str:
    if isinstance(item, tuple):
        return _text_of(item[0])
    payload = getattr(item, "payload", None)
    if payload is None and isinstance(item, dict):
        payload = item.get("payload", item)
    return (payload or {}).get("text", "") if isinstance(payload, dict) else ""


async def ingest(args) -> None:
    import cognee
    from cognee.api.v1.cognify.cognify import get_default_tasks
    from cognee.modules.pipelines.tasks.task import Task
    from cognee.modules.run_custom_pipeline import run_custom_pipeline

    question = load_questions()[args.qid]
    docs = rounds(question)
    started = time.monotonic()
    dataset = f"lme_{args.qid}"
    await cognee.add([d["text"] for d in docs], dataset_name=dataset)
    tasks = [
        task
        for task in await get_default_tasks(chunk_size=CHUNK_SIZE)
        if getattr(task, "executable", None).__name__ != "extract_graph_and_summarize"
    ]
    assert all(isinstance(task, Task) for task in tasks)
    await run_custom_pipeline(tasks=tasks, dataset=dataset, pipeline_name="chunks_only")
    meta = {
        "qid": args.qid,
        "category": category(question),
        "ingest_seconds": round(time.monotonic() - started, 1),
        "rounds": docs,
        "gold_sessions": question["answer_session_ids"],
    }
    Path(args.store, "ingest_meta.json").write_text(json.dumps(meta))
    print(f"ingested {args.qid}: {len(docs)} rounds in {meta['ingest_seconds']}s")


async def retrieve(args) -> None:
    meta = json.loads(Path(args.store, "ingest_meta.json").read_text())
    question = load_questions()[args.qid]
    by_text = {_norm(r["text"]): r["session_id"] for r in meta["rounds"]}

    def session_of(text: str):
        key = _norm(text)
        if key in by_text:
            return by_text[key]
        for known, session in by_text.items():  # long rounds are split into chunks
            if key and key in known:
                return session
        return None

    with open(args.out, "w") as out:
        for variant in args.variants.split(","):
            spec = VARIANTS[variant]
            module_name, class_name = spec["retriever"].split(":")
            retriever = getattr(importlib.import_module(module_name), class_name)(**spec["kwargs"])
            started = time.monotonic()
            retrieved = await retriever.get_retrieved_objects(question["question"])
            latency = time.monotonic() - started
            chunks = retrieved.get("chunks", []) if isinstance(retrieved, dict) else retrieved or []
            ranked, unmapped = [], 0
            for chunk in chunks:
                session = session_of(_text_of(chunk))
                if session is None:
                    unmapped += 1
                elif session not in ranked:
                    ranked.append(session)
            out.write(
                json.dumps(
                    {
                        "variant": variant,
                        "qid": args.qid,
                        "category": meta["category"],
                        "gold": meta["gold_sessions"],
                        "ranked": ranked,
                        "unmapped": unmapped,
                        "latency_s": round(latency, 4),
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest", "retrieve"])
    parser.add_argument("--qid", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--variants", default="hybrid")
    parser.add_argument("--out")
    args = parser.parse_args()
    asyncio.run(ingest(args) if args.command == "ingest" else retrieve(args))


if __name__ == "__main__":
    main()
