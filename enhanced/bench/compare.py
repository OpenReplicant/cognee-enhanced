"""Paired comparison of two variants from committed result files.

    python -m enhanced.bench.compare BASE.json:variant TREAT.json:variant \
        [--metrics recall@5,recall@10,mrr]

Questions are paired by qid; deltas are treatment - base with a 95% paired
bootstrap interval, overall and per category.
"""

import argparse
import json
from collections import defaultdict

from enhanced.bench.metrics import paired_bootstrap, scored


def _load(spec: str) -> dict:
    path, _, variant = spec.rpartition(":")
    with open(path) as handle:
        data = json.load(handle)
    rows = scored(data["variants"][variant]["questions"])
    return {r["qid"]: r for r in rows}


def compare(base: dict, treat: dict, metrics: list[str]) -> dict:
    qids = sorted(set(base) & set(treat))
    groups = defaultdict(list)
    for qid in qids:
        groups["all"].append(qid)
        groups[base[qid]["category"]].append(qid)
    report = {}
    for group, members in sorted(groups.items(), key=lambda kv: (kv[0] != "all", kv[0])):
        report[group] = {
            m: paired_bootstrap(
                [base[q]["metrics"][m] for q in members], [treat[q]["metrics"][m] for q in members]
            )
            for m in metrics
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("treatment")
    parser.add_argument("--metrics", default="recall@5,recall@10,mrr")
    args = parser.parse_args()
    metrics = args.metrics.split(",")
    report = compare(_load(args.base), _load(args.treatment), metrics)
    print(f"{args.treatment}  vs  {args.base}")
    print(f"{'group':<14}{'n':>6}  " + "  ".join(f"{m:>26}" for m in metrics))
    for group, cells in report.items():
        n = next(iter(cells.values()))["n"]
        line = "  ".join(
            f"{c['delta']:+.4f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]".rjust(26)
            for c in cells.values()
        )
        print(f"{group:<14}{n:>6}  {line}")


if __name__ == "__main__":
    main()
