"""Turn-to-graph probe: how long a chat turn takes to reach the graph.

Replays the first ``--turns`` utterances of a LoCoMo conversation through
``remember(text, session_id=...)`` at a fixed pace, with upstream's session
defaults (``CACHING``, ``AUTO_FEEDBACK`` and the automatic improve on, no
debounce). Every LLM call, embedding call, improve run and improve stage is
timed in-process, and the moment each turn becomes graph-visible is taken
from the session persist watermark, which ``cognify_session`` advances only
after the turn's ``add()`` + ``cognify()`` succeeded.

    uv run python -m enhanced.llm_proxy --port 8766 --cache <fresh dir> &
    uv run python -m enhanced.bench.turn_probe --name R0-turn-probe --turns 10

LLM calls go through the ``claude -p`` proxy, so absolute seconds include its
per-call overhead; call counts and each step's share of the time do not
depend on it.
"""

import argparse
import contextvars
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from enhanced.bench.profiles import CACHE_DIR, RESULTS_DIR, profile_env

# Upstream's defaults for a chat app that remembers each turn in a session.
SESSION_DEFAULTS = {
    "CACHING": "true",
    "AUTO_FEEDBACK": "true",
    "IMPROVE_AUTO_ENABLED": "true",
    "ENABLE_BACKEND_ACCESS_CONTROL": "true",
}

STAGE = contextvars.ContextVar("probe_stage", default="-")
EVENTS: list[dict] = []
T0 = time.monotonic()


def _now() -> float:
    return round(time.monotonic() - T0, 3)


def _record(kind: str, label: str, start: float, **extra) -> None:
    EVENTS.append(
        {"kind": kind, "label": label, "stage": STAGE.get(), "t0": start, "t1": _now(), **extra}
    )


def _timed(kind: str, label_of, fn):
    async def wrapper(*args, **kwargs):
        start = _now()
        try:
            return await fn(*args, **kwargs)
        finally:
            _record(kind, label_of(args, kwargs), start)

    return wrapper


def _llm_label(args, kwargs) -> str:
    for message in kwargs.get("messages") or []:
        if message.get("role") == "system":
            return " ".join(str(message.get("content", "")).split())[:60]
    return "(no system prompt)"


def _instrument() -> None:
    """Wrap the calls the probe times. Runs before cognee builds its LLM client."""
    import litellm

    litellm.acompletion = _timed("llm", _llm_label, litellm.acompletion)

    import cognee
    import cognee.tasks.memify.cognify_session
    from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
        FastembedEmbeddingEngine,
    )
    from cognee.modules.improve.registry import DEFAULT_STAGES

    FastembedEmbeddingEngine.embed_text = _timed(
        "embed",
        lambda args, kwargs: f"{len(args[1]) if len(args) > 1 else '?'} texts",
        FastembedEmbeddingEngine.embed_text,
    )
    cognee.add = _timed("add", lambda a, k: "add", cognee.add)
    cognee.cognify = _timed("cognify", lambda a, k: "cognify", cognee.cognify)

    for stage in DEFAULT_STAGES:

        def make(stage, run):
            async def run_timed(inputs):
                token = STAGE.set(stage.name)
                start = _now()
                result = None
                try:
                    result = await run(inputs)
                    return result
                finally:
                    _record(
                        "stage",
                        stage.name,
                        start,
                        status=getattr(result, "status", None),
                        reason=getattr(result, "reason", None),
                    )
                    STAGE.reset(token)

            return run_timed

        stage.run = make(stage, stage.run)

    # By sys.modules: the package attribute chain resolves to the function.
    improve_package = sys.modules["cognee.api.v1.improve"]
    improve_fn = improve_package.improve

    async def improve_timed(*args, **kwargs):
        start = _now()
        result = await improve_fn(*args, **kwargs)
        if getattr(result, "status", None) == "running" and hasattr(result, "wait"):
            await result.wait()
        _record(
            "improve",
            "improve",
            start,
            lock_held=bool(getattr(result, "lock_held", False)),
            rerun_requested=bool(getattr(result, "rerun_requested", False)),
            rerun_passes=len(getattr(result, "rerun_passes", None) or []),
            stages=[[s.stage, s.status, s.reason] for s in getattr(result, "stages", [])],
        )
        return result

    improve_package.improve = improve_timed

    cognify_session = sys.modules["cognee.tasks.memify.cognify_session"]
    save = cognify_session.save_persisted_qa_count

    async def save_timed(session_manager, user_id, session_id, persisted_qa_count):
        await save(session_manager, user_id, session_id, persisted_qa_count)
        EVENTS.append({"kind": "persisted", "count": persisted_qa_count, "t1": _now()})

    cognify_session.save_persisted_qa_count = save_timed


def _turns(conversation: int, n: int) -> list[str]:
    data = json.loads((CACHE_DIR / "datasets" / "locomo10.json").read_text())
    session = data[conversation]["conversation"]["session_1"]
    return [f"{turn['speaker']}: {turn['text']}" for turn in session[:n]]


async def _child(args) -> dict:
    import asyncio

    _instrument()
    import cognee

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    turns = _turns(args.conversation, args.turns)
    sent, returned = [], []
    tasks = []
    for index, text in enumerate(turns):
        due = index * args.pace
        await asyncio.sleep(max(0.0, due - _now()))
        sent.append(_now())
        result = await cognee.remember(text, dataset_name="probe", session_id="probe-session")
        returned.append(_now())
        if getattr(result, "_task", None) is not None:
            tasks.append(result._task)

    deadline = _now() + args.drain_timeout
    while _now() < deadline and _persisted() < len(turns):
        await asyncio.sleep(0.5)
    await asyncio.gather(*tasks, return_exceptions=True)
    await cognee.wait_for_background_tasks()

    from cognee.infrastructure.databases.graph import get_graph_engine

    nodes, edges = await (await get_graph_engine()).get_graph_data()
    node_types: dict[str, int] = {}
    for _, props in nodes:
        node_types[props.get("type", "?")] = node_types.get(props.get("type", "?"), 0) + 1

    visible = []
    for index in range(len(turns)):
        hit = next(
            (e["t1"] for e in EVENTS if e["kind"] == "persisted" and e["count"] > index), None
        )
        visible.append(None if hit is None else round(hit - sent[index], 2))

    return {
        "turns": turns,
        "sent": sent,
        "remember_returned_s": [round(r - s, 3) for s, r in zip(sent, returned)],
        "turn_to_graph_s": visible,
        "graph": {"nodes": len(nodes), "edges": len(edges), "node_types": node_types},
        "events": EVENTS,
    }


def _persisted() -> int:
    counts = [e["count"] for e in EVENTS if e["kind"] == "persisted"]
    return max(counts, default=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--conversation", type=int, default=0)
    parser.add_argument("--pace", type=float, default=5.0, help="seconds between turns")
    parser.add_argument("--drain-timeout", type=float, default=1200.0)
    parser.add_argument("--proxy-port", type=int, default=8766)
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE override")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    out = RESULTS_DIR / "realtime" / f"{args.name}.json"
    if args.child:
        import asyncio

        report = asyncio.run(_child(args))
        report["config"] = {k: os.environ.get(k) for k in sorted(SESSION_DEFAULTS)} | {
            "pace_s": args.pace,
            "overrides": args.env,
        }
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
        report["git"] = {"sha": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print(f"wrote {out}")
        return

    store = CACHE_DIR / "stores" / "realtime" / args.name
    env = profile_env("haiku", store)
    env.update(SESSION_DEFAULTS)
    env["LLM_ENDPOINT"] = f"http://127.0.0.1:{args.proxy_port}/v1"
    env["LOG_LEVEL"] = "WARNING"
    for override in args.env:
        key, _, value = override.partition("=")
        env[key] = value
    cmd = [sys.executable, "-m", "enhanced.bench.turn_probe", "--child", *sys.argv[1:]]
    sys.exit(
        subprocess.run(
            cmd, env=env, cwd=Path(__file__).resolve().parents[2], check=False
        ).returncode
    )


if __name__ == "__main__":
    main()
