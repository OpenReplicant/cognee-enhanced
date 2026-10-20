"""Run a retrieval benchmark and write a result file under enhanced/results/.

    python -m enhanced.bench.run --name baseline --variants hybrid,chunks,bm25 \
        [--convs 0-9] [--profile keyless] [--max-sessions N]

Stores are ingested once per (benchmark, profile, conversation) and reused;
pass a different ``--profile`` (or a store-affecting variant) to get a new one.
Each conversation x variant runs in its own subprocess under the profile env.
"""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

from enhanced.bench.metrics import aggregate
from enhanced.bench.profiles import CACHE_DIR, ENHANCED_DIR, RESULTS_DIR, profile_env, store_dir
from enhanced.bench.variants import VARIANTS

REPO = ENHANCED_DIR.parent


def _convs(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def _worker(profile: str, store: Path, extra_env: dict, *args) -> None:
    store.mkdir(parents=True, exist_ok=True)
    env = profile_env(profile, store)
    env.update(extra_env)
    cmd = [sys.executable, "-m", "enhanced.bench.locomo_worker", *args, "--store", str(store)]
    subprocess.run(cmd, env=env, cwd=REPO, check=True)


def _git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="result file name, e.g. H1-baseline")
    parser.add_argument("--variants", default="")
    parser.add_argument("--convs", default="0-9")
    parser.add_argument("--profile", default="keyless")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--ingest-only", action="store_true", help="build missing stores only")
    parser.add_argument(
        "--resume", action="store_true", help="reuse conversation x variant runs already finished"
    )
    args = parser.parse_args()

    variants = args.variants.split(",") if args.variants else []
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")

    unit_suffix = f"_s{args.max_sessions}" if args.max_sessions else ""
    run_dir = CACHE_DIR / "runs" / args.name
    if not variants and not args.ingest_only:
        parser.error("--variants is required unless --ingest-only")
    records = {v: [] for v in variants}
    ingest_seconds = {}
    for conv in _convs(args.convs):
        store = store_dir("locomo", args.profile, f"conv{conv}{unit_suffix}")
        meta_path = store / "ingest_meta.json"
        if not meta_path.exists():
            # A store without its meta is an interrupted ingest; start it over.
            shutil.rmtree(store, ignore_errors=True)
            ingest_args = ["--max-sessions", str(args.max_sessions)] if args.max_sessions else []
            _worker(args.profile, store, {}, "ingest", "--conv", str(conv), *ingest_args)
        ingest_seconds[conv] = json.loads(meta_path.read_text())["ingest_seconds"]
        if args.ingest_only:
            continue
        for variant in variants:
            out = run_dir / variant / f"conv{conv}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            done = run_dir / variant / f"conv{conv}.done"
            if args.resume and done.exists():
                records[variant].extend(json.loads(line) for line in out.read_text().splitlines())
                continue
            _worker(
                args.profile,
                store,
                VARIANTS[variant]["env"],
                "retrieve",
                "--conv",
                str(conv),
                "--variant",
                variant,
                "--out",
                str(out),
            )
            done.touch()
            records[variant].extend(json.loads(line) for line in out.read_text().splitlines())

    if args.ingest_only:
        print(f"stores ready: {ingest_seconds}")
        return

    result = {
        "suite": "locomo-retrieval",
        "name": args.name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "sha": _git("rev-parse", "HEAD"),
            # Uncommitted code changes, ignoring result files themselves.
            "dirty": bool(_git("status", "--porcelain", "--", ".", ":!enhanced/results")),
        },
        "profile": args.profile,
        "convs": _convs(args.convs),
        "max_sessions": args.max_sessions,
        "ingest_seconds": ingest_seconds,
        "variants": {
            v: {
                "spec": VARIANTS[v],
                "metrics": aggregate(records[v]),
                "questions": [
                    {k: r[k] for k in ("qid", "category", "gold", "ranked")} for r in records[v]
                ],
            }
            for v in variants
        },
    }
    out_path = RESULTS_DIR / "locomo" / f"{args.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_path.relative_to(REPO)}")
    for v in variants:
        m = result["variants"][v]["metrics"]["all"]
        print(
            f"  {v:<24} n={m['n']} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} "
            f"MRR={m['mrr']:.3f} p50={m.get('latency_p50_s')}s"
        )


if __name__ == "__main__":
    main()
