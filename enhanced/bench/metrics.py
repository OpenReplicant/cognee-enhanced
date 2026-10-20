"""Retrieval metrics over per-question records ``{"gold": [...], "ranked": [...]}``.

Questions without gold evidence are excluded from every metric.
"""

import random
from collections import defaultdict

KS = (1, 3, 5, 10, 20)


def question_metrics(gold: list, ranked: list) -> dict:
    gold_set = set(gold)
    metrics = {}
    for k in KS:
        top = set(ranked[:k])
        metrics[f"recall@{k}"] = len(gold_set & top) / len(gold_set)
        metrics[f"all@{k}"] = float(gold_set <= top)
    first = next((rank for rank, item in enumerate(ranked, 1) if item in gold_set), None)
    metrics["mrr"] = 1.0 / first if first else 0.0
    return metrics


def scored(records: list[dict]) -> list[dict]:
    return [
        {**r, "metrics": question_metrics(r["gold"], r["ranked"])} for r in records if r["gold"]
    ]


def aggregate(records: list[dict]) -> dict:
    rows = scored(records)
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["category"]].append(row)
    out = {}
    for name, members in sorted(groups.items()):
        keys = members[0]["metrics"].keys()
        out[name] = {"n": len(members)}
        out[name].update(
            {k: round(sum(m["metrics"][k] for m in members) / len(members), 4) for k in keys}
        )
    latencies = sorted(r["latency_s"] for r in records if "latency_s" in r)
    if latencies:
        out["all"]["latency_p50_s"] = round(latencies[len(latencies) // 2], 4)
        out["all"]["latency_p95_s"] = round(latencies[int(len(latencies) * 0.95)], 4)
    return out


def paired_bootstrap(a: list[float], b: list[float], iterations: int = 10000, seed: int = 0):
    """Mean of b - a with a 95% percentile interval, resampling questions."""
    assert len(a) == len(b) and a, "paired samples must align and be non-empty"
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return {
        "delta": round(sum(diffs) / n, 4),
        "ci95": [
            round(means[int(0.025 * iterations)], 4),
            round(means[int(0.975 * iterations)], 4),
        ],
        "n": n,
        "wins": sum(d > 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
    }
