"""Attribute QA errors to retrieval or to the reader.

    python -m enhanced.bench.attribute QA.json:variant RETRIEVAL.json:variant --k 20

For each judged question, the retrieval run's ranked windows for the same
retriever (the chunk lane is deterministic, so they are the windows the reader
saw) tell whether the gold windows were in the top ``k``. Wrong answers then
split into: retrieval miss (no gold window in context), partial (some but not
all), reader error (all gold windows in context). Passages only: evidence the
entity and fact lines carried is not credited.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load(spec: str) -> dict:
    path, _, variant = spec.rpartition(":")
    return {
        q["qid"]: q for q in json.loads(Path(path).read_text())["variants"][variant]["questions"]
    }


def attribute(qa: dict, retrieval: dict, k: int) -> dict:
    table = defaultdict(Counter)
    for qid, row in qa.items():
        if row.get("correct") is None or qid not in retrieval:
            continue
        gold = set(retrieval[qid]["gold"])
        shown = set(retrieval[qid]["ranked"][:k])
        if not gold:
            where = "no_evidence"
        elif gold <= shown:
            where = "all_gold_shown"
        elif gold & shown:
            where = "some_gold_shown"
        else:
            where = "no_gold_shown"
        outcome = "correct" if row["correct"] else "wrong"
        for group in ("all", row["category"]):
            table[group][f"{outcome}:{where}"] += 1
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("qa")
    parser.add_argument("retrieval")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    table = attribute(_load(args.qa), _load(args.retrieval), args.k)
    cols = ["no_gold_shown", "some_gold_shown", "all_gold_shown"]
    print(f"{'group':<12} {'n':>4} {'wrong':>6} | wrong by evidence shown: " + "  ".join(cols))
    for group in sorted(table, key=lambda g: (g != "all", g)):
        c = table[group]
        n = sum(c.values())
        wrong = sum(v for key, v in c.items() if key.startswith("wrong"))
        cells = "  ".join(f"{c[f'wrong:{col}']:>{len(col)}}" for col in cols)
        print(f"{group:<12} {n:>4} {wrong:>6} | {' ' * 25}{cells}")


if __name__ == "__main__":
    main()
