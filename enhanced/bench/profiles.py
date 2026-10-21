"""Ingestion/runtime profiles: the env a benchmark process runs under.

A profile is applied before cognee is imported (cognee caches its config at
import), so every ingest and retrieval run happens in its own subprocess.
"""

import contextlib
import fcntl
import os
from pathlib import Path

ENHANCED_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.getenv("ENHANCED_CACHE_DIR", ENHANCED_DIR / ".cache"))
RESULTS_DIR = ENHANCED_DIR / "results"

# Variables that leak in from the host and change cognee's behaviour
# (AWS keys switch ingestion to S3, an LLM key switches the extractor).
_SCRUB = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
)

_COMMON = {
    "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    "CACHING": "false",
    "AUTO_FEEDBACK": "false",
    "TELEMETRY_DISABLED": "1",
    "LOG_LEVEL": "ERROR",
    "COGNEE_LOG_FILE": "false",
    "IMPROVE_AUTO_ENABLED": "false",
}

# The answering LLM for QA runs (T1): Claude via the local proxy. Applied on
# top of any profile, so answers on a keyless store and on a haiku store come
# from the same model; it is read only at query time.
ANSWER_ENV = {
    "LLM_PROVIDER": "custom",
    "LLM_MODEL": "openai/haiku",
    "LLM_ENDPOINT": "http://127.0.0.1:8765/v1",
    "LLM_API_KEY": "local-proxy",
    "LLM_INSTRUCTOR_MODE": "json_mode",
}

PROFILES = {
    # Fully local, no network after model download: GLiNER graph + bge-small.
    # Deterministic and free; the graph is weaker than an LLM-built one, so
    # graph-side hypotheses validated here still need an `llm` profile check.
    "keyless": {
        "GRAPH_EXTRACTOR": "gliner_demo",
        "EMBEDDING_PROVIDER": "fastembed",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "EMBEDDING_DIMENSIONS": "384",
    },
    # Chunks only (LongMemEval): the store is built without graph extraction,
    # so only the local embeddings matter; same settings as keyless.
    "chunks": {
        "GRAPH_EXTRACTOR": "gliner_demo",
        "EMBEDDING_PROVIDER": "fastembed",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "EMBEDDING_DIMENSIONS": "384",
    },
    # LLM-built graph and summaries from Claude Haiku through the local
    # Claude Code proxy (`python -m enhanced.llm_proxy --model haiku`, port
    # 8765); same local embeddings as keyless, so the two profiles differ
    # only in who extracts the graph and writes the summaries.
    "haiku": {
        "GRAPH_EXTRACTOR": "llm",
        "LLM_PROVIDER": "custom",
        "LLM_MODEL": "openai/haiku",
        "LLM_ENDPOINT": "http://127.0.0.1:8765/v1",
        "LLM_API_KEY": "local-proxy",
        "LLM_INSTRUCTOR_MODE": "json_mode",
        "EMBEDDING_PROVIDER": "fastembed",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "EMBEDDING_DIMENSIONS": "384",
    },
}


def profile_env(profile: str, store_dir: Path) -> dict:
    """The environment for a subprocess running under ``profile`` on ``store_dir``."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB}
    env.update(_COMMON)
    env.update(PROFILES[profile])
    env["DATA_ROOT_DIRECTORY"] = str(store_dir / "data")
    env["SYSTEM_ROOT_DIRECTORY"] = str(store_dir / "system")
    return env


def store_dir(benchmark: str, profile: str, unit: str) -> Path:
    """Where the ingested store for one benchmark unit (e.g. a conversation) lives."""
    return CACHE_DIR / "stores" / benchmark / profile / unit


@contextlib.contextmanager
def store_lock(store: Path):
    """Hold an exclusive lock on ``store`` while a subprocess uses it.

    The embedded graph database takes an exclusive file lock, so two benchmark
    jobs opening the same store at once crash one of them. Every harness
    entry point that launches a store subprocess goes through this, so jobs
    wait for each other instead.
    """
    store.mkdir(parents=True, exist_ok=True)
    with open(store.parent / f".{store.name}.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
