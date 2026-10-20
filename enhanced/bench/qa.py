"""LoCoMo answer accuracy (tier T1): answer with each variant, grade with a judge.

    python -m enhanced.bench.qa --name H1-qa --variants hybrid,hybrid_bm25 \\
        [--convs 5-9] [--profile keyless] [--per-conv 30] [--seed 0]

Needs the proxy (``python -m enhanced.llm_proxy --model haiku``) running: it
answers (through each variant's own completion path, upstream's
``qa_prompts/unified.txt``) and judges (upstream's LoCoMo judge prompt).
Questions are a fixed per-conversation sample, excluding open_domain and the
gold answers upstream's audit marks as corrupted, so every variant answers the
same questions. Judge verdicts are cached by the proxy like any other call.
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict

from cognee.eval_framework.benchmark_adapters.locomo_adapter import LocomoAdapter
from cognee.eval_framework.locomo.metrics.llm_judge import (
    _PROMPT_PATH as JUDGE_PROMPT_PATH,
)
from cognee.eval_framework.locomo.metrics.llm_judge import (
    parse_judge_output,
    render_judge_prompt,
)
from enhanced.bench.locomo_worker import QA_PROMPT_PATH
from enhanced.bench.metrics import paired_bootstrap
from enhanced.bench.profiles import (
    ANSWER_ENV,
    CACHE_DIR,
    RESULTS_DIR,
    profile_env,
    store_dir,
    store_lock,
)
from enhanced.bench.run import REPO, _convs, _git
from enhanced.bench.variants import VARIANTS

EXCLUSIONS_PATH = REPO / "cognee/eval_framework/locomo/report_artifacts/gold_exclusions.json"
JUDGE_SYSTEM = "You are a strict but fair grader. Reply with JSON only."


def _excluded() -> set[str]:
    data = json.loads(EXCLUSIONS_PATH.read_text())
    return {
        f"conv{q['conversation_index']}:{q['question_idx']}"
        for group in data["sets"].values()
        for q in group["questions"]
    }


def sample_qids(convs: list[int], per_conv: int, seed: int) -> dict[int, list[str]]:
    adapter = LocomoAdapter(data_path=str(CACHE_DIR / "datasets" / "locomo10.json"))
    excluded = _excluded()
    picked = {}
    for conv in convs:
        pool = [
            f"conv{conv}:{q['question_idx']}"
            for q in adapter.questions_for(adapter.load_conversation(conv))
            if q["question_type"] != "open_domain"
        ]
        pool = [qid for qid in pool if qid not in excluded]
        picked[conv] = sorted(
            random.Random(seed * 1000 + conv).sample(pool, min(per_conv, len(pool)))
        )
    return picked


async def judge_all(rows: list[dict]) -> None:
    import litellm

    semaphore = asyncio.Semaphore(4)

    async def judge(row):
        prompt = render_judge_prompt(
            question=row["question"],
            gold_answer=row["gold"],
            model_answer=row["answer"],
            question_type=row["category"],
            adversarial_answer=row.get("adversarial_answer"),
        )
        async with semaphore:
            for _ in range(2):
                response = await litellm.acompletion(
                    model=ANSWER_ENV["LLM_MODEL"],
                    api_base=ANSWER_ENV["LLM_ENDPOINT"],
                    api_key=ANSWER_ENV["LLM_API_KEY"],
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                )
                verdict = parse_judge_output(response.choices[0].message.content or "")
                if verdict["label"]:
                    break
        row["label"] = verdict["label"]
        row["judge_reason"] = verdict["reason"]
        row["correct"] = None if verdict["label"] is None else float(verdict["label"] == "CORRECT")

    await asyncio.gather(*(judge(row) for row in rows))


def _accuracy(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row["correct"] is not None:
            groups["all"].append(row["correct"])
            groups[row["category"]].append(row["correct"])
    out = {
        g: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for g, v in sorted(groups.items())
    }
    out["all"]["unscored"] = sum(row["correct"] is None for row in rows)
    return out


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--convs", default="5-9")
    parser.add_argument("--profile", default="keyless")
    parser.add_argument("--per-conv", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    variants = args.variants.split(",")
    convs = _convs(args.convs)

    qids = sample_qids(convs, args.per_conv, args.seed)
    run_dir = CACHE_DIR / "runs" / args.name / "qa"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = {v: [] for v in variants}
    for conv in convs:
        store = store_dir("locomo", args.profile, f"conv{conv}")
        if not (store / "ingest_meta.json").exists():
            sys.exit(f"missing store {store}; ingest it with enhanced.bench.run --ingest-only")
        qids_file = run_dir / f"qids_conv{conv}.json"
        qids_file.write_text(json.dumps(qids[conv]))
        for variant in variants:
            out = run_dir / variant / f"conv{conv}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            env = profile_env(args.profile, store)
            env.update(ANSWER_ENV)
            env.update(VARIANTS[variant]["env"])
            with store_lock(store):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "enhanced.bench.locomo_worker",
                        "qa",
                        "--conv",
                        str(conv),
                        "--store",
                        str(store),
                        "--variant",
                        variant,
                        "--qids",
                        str(qids_file),
                        "--out",
                        str(out),
                    ],
                    env=env,
                    cwd=REPO,
                    check=True,
                )
            rows[variant].extend(json.loads(line) for line in out.read_text().splitlines())

    for variant in variants:
        asyncio.run(judge_all(rows[variant]))

    result = {
        "suite": "locomo-qa",
        "name": args.name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "sha": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--", ".", ":!enhanced/results")),
        },
        "profile": args.profile,
        "convs": convs,
        "sample": {
            "per_conv": args.per_conv,
            "seed": args.seed,
            "excludes": "open_domain + upstream gold_exclusions.json",
        },
        "answer_model": "claude haiku via enhanced.llm_proxy",
        "judge_model": "claude haiku via enhanced.llm_proxy",
        "answer_prompt_sha": _sha(QA_PROMPT_PATH),
        "judge_prompt_sha": _sha(JUDGE_PROMPT_PATH),
        "variants": {
            v: {"spec": VARIANTS[v], "metrics": _accuracy(rows[v]), "questions": rows[v]}
            for v in variants
        },
    }
    out_path = RESULTS_DIR / "locomo-qa" / f"{args.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_path.relative_to(REPO)}")
    base = {r["qid"]: r for r in rows[variants[0]]}
    for v in variants:
        m = result["variants"][v]["metrics"]["all"]
        line = f"  {v:<24} n={m['n']} accuracy={m['accuracy']:.3f} unscored={m['unscored']}"
        if v != variants[0]:
            pairs = [
                (base[r["qid"]]["correct"], r["correct"])
                for r in rows[v]
                if r["qid"] in base
                and r["correct"] is not None
                and base[r["qid"]]["correct"] is not None
            ]
            cmp = paired_bootstrap([a for a, _ in pairs], [b for _, b in pairs])
            line += f"  vs {variants[0]}: {cmp['delta']:+.3f} [{cmp['ci95'][0]:+.3f},{cmp['ci95'][1]:+.3f}]"
        print(line)


if __name__ == "__main__":
    main()
