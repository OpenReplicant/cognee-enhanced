"""OpenAI-compatible chat endpoint backed by headless Claude Code (`claude -p`).

Lets cognee (via litellm, provider ``custom``) use a Claude model through the
local Claude Code login when no API key is available:

    uv run python -m enhanced.llm_proxy --port 8765 --model haiku

Every completion is cached on disk by a hash of (model, messages), so a
benchmark rerun replays the same answers for free and reproducibly. Usage
counters are appended to ``<cache>/usage.jsonl``.

Each call runs in an empty directory with tools, settings, MCP servers and
session persistence off, so the request is just the system prompt + the
flattened conversation.
"""

import argparse
import asyncio
import hashlib
import json
import re
import tempfile
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from enhanced.bench.profiles import CACHE_DIR

app = FastAPI()
STATE: dict = {}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def _flatten(messages: list[dict]) -> tuple[str, str]:
    """System prompt, and the remaining turns as one prompt."""
    system = "\n\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) == 1:
        return system, _text(turns[0].get("content"))
    prompt = "\n\n".join(f"[{m.get('role', 'user')}]\n{_text(m.get('content'))}" for m in turns)
    return system, prompt


def _strip_fences(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.DOTALL)
    return match.group(1) if match else text


async def _claude(system: str, prompt: str, wants_json: bool) -> dict:
    if wants_json:
        system += "\n\nRespond with a single JSON object only: no prose, no code fences."
    cmd = [
        STATE["claude_bin"],
        "-p",
        "--model",
        STATE["model"],
        "--output-format",
        "json",
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt",
        system or "You are a helpful assistant.",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=STATE["workdir"],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), STATE["timeout"])
    except asyncio.TimeoutError:
        proc.kill()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {err.decode()[-500:]}")
    result = json.loads(out)
    if result.get("is_error"):
        raise RuntimeError(f"claude error: {str(result.get('result'))[:500]}")
    return result


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if body.get("stream"):
        raise HTTPException(400, "streaming is not supported")
    messages = body.get("messages", [])
    response_format = body.get("response_format") or {}
    wants_json = response_format.get("type") in ("json_object", "json_schema")
    system, prompt = _flatten(messages)

    key = hashlib.sha256(
        json.dumps([STATE["model"], system, prompt, wants_json], sort_keys=True).encode()
    ).hexdigest()
    cache_file = STATE["cache"] / key[:2] / f"{key}.json"
    cached = cache_file.exists()
    if cached:
        result = json.loads(cache_file.read_text())
    else:
        async with STATE["semaphore"]:
            started = time.monotonic()
            last_error = None
            for _ in range(STATE["retries"] + 1):
                try:
                    result = await _claude(system, prompt, wants_json)
                    break
                except (RuntimeError, OSError, ValueError, asyncio.TimeoutError) as error:
                    last_error = error
            else:
                raise HTTPException(502, f"claude call failed: {last_error}")
            result["_wall_s"] = round(time.monotonic() - started, 2)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result))

    usage = result.get("usage", {})
    prompt_tokens = (
        usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
    )
    completion_tokens = usage.get("output_tokens", 0)
    with open(STATE["cache"] / "usage.jsonl", "a") as log:
        log.write(
            json.dumps(
                {
                    "key": key,
                    "cached": cached,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "wall_s": result.get("_wall_s"),
                    "cost_usd_list": result.get("total_cost_usd"),
                }
            )
            + "\n"
        )

    content = result.get("result", "")
    if wants_json:
        content = _strip_fences(content)
    model_used = next(iter(result.get("modelUsage", {}) or {}), STATE["model"])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": STATE["model"], "object": "model"}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--cache", default=str(CACHE_DIR / "llm_proxy"))
    args = parser.parse_args()
    cache = Path(args.cache) / args.model
    cache.mkdir(parents=True, exist_ok=True)
    STATE.update(
        model=args.model,
        cache=cache,
        timeout=args.timeout,
        retries=args.retries,
        claude_bin=args.claude_bin,
        semaphore=asyncio.Semaphore(args.concurrency),
        workdir=tempfile.mkdtemp(prefix="llm-proxy-"),
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
