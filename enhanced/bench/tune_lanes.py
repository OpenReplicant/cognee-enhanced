"""Tune the hybrid chunk lane's RRF weights offline (enhanced/HYPOTHESES.md H2).

    python -m enhanced.bench.tune_lanes dump     --profile haiku --convs 0-4
    python -m enhanced.bench.tune_lanes validate --profile haiku --convs 5 \\
        --against enhanced/results/locomo/H1-test-haiku.json:hybrid_bm25
    python -m enhanced.bench.tune_lanes tune     --profile haiku --convs 0-4

``dump`` records every question's chunk-lane candidates with their dense,
summary and lexical ranks (one retrieval pass). ``fuse`` re-ranks them with
the real ``rank_chunk_summary_pairs``, so a weight grid costs milliseconds
instead of a retrieval pass per point. ``validate`` checks that equal-weight
offline fusion reproduces a committed ``hybrid_bm25`` run exactly before any
tuned weight is trusted. The tuned weights are then run through the real
retriever on the test split for the decision.
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs
from enhanced.bench.metrics import question_metrics
from enhanced.bench.profiles import CACHE_DIR, RESULTS_DIR, profile_env, store_dir, store_lock
from enhanced.bench.run import REPO, _convs, _git
from enhanced.bench.variants import DEPTH

GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _dump_path(profile: str, conv: int):
    return CACHE_DIR / "runs" / f"lanes-{profile}" / f"conv{conv}.jsonl"


def dump(profile: str, convs: list[int]) -> None:
    for conv in convs:
        out = _dump_path(profile, conv)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        store = store_dir("locomo", profile, f"conv{conv}")
        cmd = [sys.executable, "-m", "enhanced.bench.locomo_worker", "lanes", "--conv", str(conv)]
        cmd += ["--store", str(store), "--out", str(out) + ".tmp"]
        with store_lock(store):
            subprocess.run(cmd, env=profile_env(profile, store), cwd=REPO, check=True)
        (out.parent / (out.name + ".tmp")).rename(out)


def load(profile: str, convs: list[int]) -> list[dict]:
    rows = []
    for conv in convs:
        rows.extend(json.loads(line) for line in _dump_path(profile, conv).read_text().splitlines())
    return rows


def fuse(candidates: list[dict], weights, depth: int = DEPTH) -> list[int]:
    """Ranked windows from the real chunk-lane fusion at ``weights``."""
    pairs = [
        {
            "chunk": {"id": c["id"], "importance_weight": c["imp"]},
            "chunk_id": c["id"],
            "vector_rank": c["v"],
            "summary_rank": c["s"],
            "lexical_rank": c["l"],
        }
        for c in candidates
    ]
    window = {c["id"]: c["window"] for c in candidates}
    ranked = []
    for pair in rank_chunk_summary_pairs(
        pairs, depth, use_importance_weight=True, lane_weights=tuple(weights)
    ):
        w = window.get(pair["chunk_id"])
        if w is not None and w not in ranked:
            ranked.append(w)
    return ranked


def category_scores(rows: list[dict], weights, metric: str = "recall@5") -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        if not r["gold"]:
            continue
        value = question_metrics(r["gold"], fuse(r["candidates"], weights))[metric]
        groups.setdefault("all", []).append(value)
        groups.setdefault(r["category"], []).append(value)
    return {k: sum(v) / len(v) for k, v in groups.items()}


def score(rows: list[dict], weights, metric: str = "recall@5") -> float:
    return category_scores(rows, weights, metric)["all"]


def validate(profile: str, convs: list[int], against: str) -> int:
    path, _, variant = against.rpartition(":")
    reference = {
        q["qid"]: q["ranked"]
        for q in json.loads(Path(path).read_text())["variants"][variant]["questions"]
    }
    rows = load(profile, convs)
    mismatched = [
        r["qid"] for r in rows if fuse(r["candidates"], (1, 1, 1)) != reference.get(r["qid"])
    ]
    print(f"validate: {len(rows) - len(mismatched)}/{len(rows)} rankings reproduced")
    for qid in mismatched[:5]:
        print("  mismatch", qid)
    return len(mismatched)


def tune(profile: str, convs: list[int], metric: str, max_category_drop: float | None) -> None:
    """Grid search; with ``max_category_drop``, weights that lower any category's
    dev score by more than that (vs equal weights) are not eligible."""
    rows = load(profile, convs)
    equal_scores = category_scores(rows, (1.0, 1.0, 1.0), metric)
    results = []
    for dense, summary in itertools.product(GRID, GRID):
        weights = (dense, summary, 1.0)  # RRF order is scale-invariant: fix lexical at 1
        scores = category_scores(rows, weights, metric)
        worst = min(scores[c] - equal_scores[c] for c in scores if c != "all")
        results.append(
            {
                "weights": weights,
                metric: round(scores["all"], 4),
                "worst_category_delta": round(worst, 4),
                "eligible": max_category_drop is None or worst >= -max_category_drop,
            }
        )
    results.sort(key=lambda r: (not r["eligible"], -r[metric]))
    equal = next(r for r in results if r["weights"] == (1.0, 1.0, 1.0))
    suffix = "" if max_category_drop is None else f"-maxdrop{max_category_drop:g}"
    out = {
        "suite": "locomo-lane-tuning",
        "profile": profile,
        "convs": convs,
        "metric": metric,
        "max_category_drop": max_category_drop,
        "git": {"sha": _git("rev-parse", "HEAD")},
        "grid": list(GRID),
        "equal_weights": equal,
        "best": results[0],
        "top10": results[:10],
        "all": results,
    }
    path = RESULTS_DIR / "locomo" / f"H2-tune-{profile}{suffix}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path.relative_to(REPO)}")
    print(f"  equal weights {equal}")
    for r in results[:10]:
        print(f"  {r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["dump", "validate", "tune"])
    parser.add_argument("--profile", default="haiku")
    parser.add_argument("--convs", default="0-4")
    parser.add_argument("--against", help="validate: RESULT.json:variant to reproduce")
    parser.add_argument("--metric", default="recall@5")
    parser.add_argument(
        "--max-category-drop",
        type=float,
        default=None,
        help="tune: exclude weights that lower any category's dev score by more than this",
    )
    args = parser.parse_args()
    convs = _convs(args.convs)
    if args.command == "dump":
        dump(args.profile, convs)
    elif args.command == "validate":
        dump(args.profile, convs)
        sys.exit(1 if validate(args.profile, convs, args.against) else 0)
    else:
        dump(args.profile, convs)
        tune(args.profile, convs, args.metric, args.max_category_drop)


if __name__ == "__main__":
    main()
