"""LongMemEval-S retrieval benchmark (tier T0b): session recall on chunks-only stores.

    python -m enhanced.bench.lme_run --name lme-baseline --variants hybrid,hybrid_bm25 \\
        [--per-category 10] [--seed 0]

A fixed stratified sample (``--per-category`` questions from each of the six
question types plus abstention), one chunks-only store per question (see
``longmemeval_worker``), all variants retrieved in one process per store.
Writes ``results/longmemeval/<name>.json`` in the LoCoMo result format, so
``bench.compare`` works unchanged (gold and ranked are session ids).
"""

import argparse
import datetime
import json
import random
import subprocess
import sys
from collections import defaultdict

from enhanced.bench.longmemeval_worker import category, load_questions
from enhanced.bench.metrics import aggregate
from enhanced.bench.profiles import CACHE_DIR, RESULTS_DIR, profile_env, store_dir, store_lock
from enhanced.bench.run import REPO, _git
from enhanced.bench.variants import VARIANTS

PROFILE = "chunks"


def sample(per_category: int, seed: int) -> list[str]:
    by_category = defaultdict(list)
    for qid, question in sorted(load_questions().items()):
        by_category[category(question)].append(qid)
    picked = []
    for name, qids in sorted(by_category.items()):
        picked += sorted(random.Random(f"{seed}:{name}").sample(qids, min(per_category, len(qids))))
    return picked


def _worker(store, *args) -> None:
    cmd = [sys.executable, "-m", "enhanced.bench.longmemeval_worker", *args, "--store", str(store)]
    with store_lock(store):
        subprocess.run(cmd, env=profile_env(PROFILE, store), cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ingest-only", action="store_true")
    args = parser.parse_args()
    variants = args.variants.split(",")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants {unknown}")

    qids = sample(args.per_category, args.seed)
    run_dir = CACHE_DIR / "runs" / f"lme-{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records = {v: [] for v in variants}
    for qid in qids:
        store = store_dir("longmemeval", PROFILE, qid)
        if not (store / "ingest_meta.json").exists():
            _worker(store, "ingest", "--qid", qid)
        if args.ingest_only:
            continue
        out = run_dir / f"{qid}.jsonl"
        if not out.exists():
            tmp = run_dir / f"{qid}.jsonl.tmp"
            _worker(
                store, "retrieve", "--qid", qid, "--variants", ",".join(variants), "--out", str(tmp)
            )
            tmp.rename(out)
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if row["variant"] in records:
                records[row["variant"]].append(row)
    if args.ingest_only:
        print(f"stores ready for {len(qids)} questions")
        return

    result = {
        "suite": "longmemeval-s-retrieval",
        "name": args.name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "sha": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--", ".", ":!enhanced/results")),
        },
        "profile": PROFILE,
        "sample": {"per_category": args.per_category, "seed": args.seed, "qids": qids},
        "unit": "session (gold = answer_session_ids)",
        "variants": {
            v: {
                "spec": VARIANTS[v],
                "metrics": aggregate(records[v]),
                "unmapped_chunks": sum(r["unmapped"] for r in records[v]),
                "questions": [
                    {k: r[k] for k in ("qid", "category", "gold", "ranked")} for r in records[v]
                ],
            }
            for v in variants
        },
    }
    out_path = RESULTS_DIR / "longmemeval" / f"{args.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_path.relative_to(REPO)}")
    for v in variants:
        m = result["variants"][v]["metrics"]["all"]
        print(
            f"  {v:<26} n={m['n']} R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} "
            f"all@5={m['all@5']:.3f} MRR={m['mrr']:.3f} unmapped={result['variants'][v]['unmapped_chunks']}"
        )


if __name__ == "__main__":
    main()
