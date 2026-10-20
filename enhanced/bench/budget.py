"""Evidence recall at a fixed context-token budget (enhanced/HYPOTHESES.md H4).

    python -m enhanced.bench.budget enhanced/results/locomo/H1-test-haiku.json \\
        [--budgets 1000,2000,4000]

Works offline on any committed retrieval result: windows are taken in rank
order while they fit the budget (a window that would overflow it ends the
context), with token counts from the stores' ``ingest_meta.json``
(tiktoken ``cl100k_base``). Counts cover passage text only: the entity and
fact lines hybrid adds to its prompt are not included, so this compares the
chunk lane's ordering at equal passage cost.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import tiktoken

from enhanced.bench.profiles import store_dir


def window_tokens(profile: str, conv: int, max_sessions) -> dict[int, int]:
    suffix = f"_s{max_sessions}" if max_sessions else ""
    meta = json.loads(
        (store_dir("locomo", profile, f"conv{conv}{suffix}") / "ingest_meta.json").read_text()
    )
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = {i: len(encoding.encode(w["text"])) for i, w in enumerate(meta["windows"])}
    tokens[-1] = len(encoding.encode(meta["overview_text"]))
    return tokens


def within_budget(ranked: list[int], tokens: dict[int, int], budget: int) -> tuple[list[int], int]:
    kept, used = [], 0
    for window in ranked:
        cost = tokens.get(window, 0)
        if used + cost > budget:
            break
        kept.append(window)
        used += cost
    return kept, used


def report(result_path: str, budgets: list[int]) -> dict:
    data = json.loads(Path(result_path).read_text())
    tokens = {c: window_tokens(data["profile"], c, data.get("max_sessions")) for c in data["convs"]}
    out = {}
    for variant, body in data["variants"].items():
        sums = defaultdict(float)
        n = 0
        for q in body["questions"]:
            if not q["gold"]:
                continue
            n += 1
            conv = int(q["qid"].split(":")[0].removeprefix("conv"))
            gold = set(q["gold"])
            for budget in budgets:
                kept, used = within_budget(q["ranked"], tokens[conv], budget)
                sums[f"recall@{budget}t"] += len(gold & set(kept)) / len(gold)
                sums[f"tokens@{budget}t"] += used
        out[variant] = {"n": n, **{k: round(v / n, 4) for k, v in sums.items()}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--budgets", default="1000,2000,4000")
    args = parser.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    for variant, metrics in report(args.result, budgets).items():
        cells = "  ".join(
            f"R@{b}t={metrics[f'recall@{b}t']:.3f} ({metrics[f'tokens@{b}t']:.0f})" for b in budgets
        )
        print(f"{variant:<24} n={metrics['n']}  {cells}")


if __name__ == "__main__":
    main()
