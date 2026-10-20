"""Entity-duplication baseline and labelled alias pairs (enhanced/HYPOTHESES.md H9).

    python -m enhanced.bench.entity_dups dump  --profile haiku --convs 5-9
    python -m enhanced.bench.entity_dups label --profile haiku --convs 5-9

``dump`` (one subprocess per store, read-only) writes each store's Entity
nodes (name, description, type) and relationship-name counts. ``label``
blocks candidate pairs with three cheap signals (name containment, content
token overlap, name+description embedding similarity), has Claude Haiku via
the proxy adjudicate them in batches, and writes the labelled pairs (the H9
evaluation set) and a baseline report: how many entities have a confirmed
duplicate, and each blocking signal's precision. Labels are an LLM judgment
on names and descriptions, not ground truth; the report says so.
"""

import argparse
import asyncio
import itertools
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from enhanced.bench.profiles import ANSWER_ENV, CACHE_DIR, RESULTS_DIR, profile_env, store_dir
from enhanced.bench.run import REPO, _convs, _git

STOP = {"the", "a", "an", "of", "and", "s", "in", "on", "at", "to", "for", "with", "his", "her"}
EMBED_THRESHOLD = 0.93
BATCH = 25

ADJUDICATE_PROMPT = """Below are pairs of entity nodes extracted from ONE conversation between \
two people. For each pair decide whether both nodes refer to the same real-world thing (the \
same person, pet, place, object, event or activity), so that a knowledge graph should merge \
them. Different instances, a part versus the whole, a person versus something they own, and \
a general concept versus a specific instance are NOT the same.

{pairs}

Reply with JSON only: {{"labels": [{{"pair": <number>, "same": true or false}}, ...]}} \
covering every pair."""


def _dump_path(profile: str, conv: int) -> Path:
    return CACHE_DIR / "runs" / f"entities-{profile}" / f"conv{conv}.json"


DUMP_SCRIPT = """
import asyncio, json, sys
from collections import Counter
from cognee.infrastructure.databases.graph import get_graph_engine

async def main(out):
    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    by_id = {str(i): d for i, d in nodes}
    type_of = {}
    for source, target, rel, _ in edges:
        if rel == "is_a" and by_id.get(str(target), {}).get("type") == "EntityType":
            type_of[str(source)] = by_id[str(target)].get("name")
    entities = [
        {"id": str(i), "name": d.get("name") or "", "description": d.get("description") or "",
         "type": type_of.get(str(i))}
        for i, d in nodes if d.get("type") == "Entity"
    ]
    relations = Counter(rel for _, _, rel, _ in edges)
    json.dump({"entities": entities, "relations": relations}, open(out, "w"))

asyncio.run(main(sys.argv[1]))
"""


def dump(profile: str, convs: list[int]) -> None:
    for conv in convs:
        out = _dump_path(profile, conv)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        store = store_dir("locomo", profile, f"conv{conv}")
        subprocess.run(
            [sys.executable, "-c", DUMP_SCRIPT, str(out)],
            env=profile_env(profile, store),
            cwd=REPO,
            check=True,
        )


def _tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", name.lower()) if t not in STOP}


def candidate_pairs(entities: list[dict], embeddings) -> list[dict]:
    """Pairs flagged by at least one blocking signal, with the signals that fired.

    containment: one name's content tokens are a subset of the other's and
    cover at least half of it. token_jaccard: content-token Jaccard >= 0.5.
    Names whose numbers differ are never paired. embedding: mutual nearest
    neighbours of the same type with cosine >=
    EMBED_THRESHOLD (short names alone sit close together in bge-small, so a
    plain threshold flags every pair of person names).
    """
    import numpy as np

    vectors = np.asarray(embeddings)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -1.0)
    nearest = similarity.argmax(axis=1)
    tokens = [_tokens(e["name"]) for e in entities]
    pairs = []
    for i, j in itertools.combinations(range(len(entities)), 2):
        a, b = entities[i], entities[j]
        if not a["name"] or not b["name"] or a["name"] == b["name"]:
            continue
        ta, tb = tokens[i], tokens[j]
        numbers_a = {t for t in ta if t.isdigit()}
        numbers_b = {t for t in tb if t.isdigit()}
        if numbers_a and numbers_b and numbers_a != numbers_b:
            continue  # "session 4" vs "session 6", "2022" vs "2023"
        signals = []
        small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        if small and small <= large and len(small) / len(large) >= 0.5:
            signals.append("containment")
        if ta and tb and len(ta & tb) / len(ta | tb) >= 0.5:
            signals.append("token_jaccard")
        if (
            nearest[i] == j
            and nearest[j] == i
            and a["type"] == b["type"]
            and similarity[i, j] >= EMBED_THRESHOLD
        ):
            signals.append("embedding")
        if signals:
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "signals": signals,
                    "cosine": round(float(similarity[i, j]), 4),
                }
            )
    return pairs


async def adjudicate(pairs: list[dict]) -> None:
    import litellm

    semaphore = asyncio.Semaphore(4)

    def describe(e):
        return f'"{e["name"]}" (type: {e["type"] or "?"}; {e["description"][:160]})'

    async def run(batch):
        listing = "\n".join(
            f"{n}. {describe(p['a'])}  vs  {describe(p['b'])}" for n, p in enumerate(batch, 1)
        )
        async with semaphore:
            response = await litellm.acompletion(
                model=ANSWER_ENV["LLM_MODEL"],
                api_base=ANSWER_ENV["LLM_ENDPOINT"],
                api_key=ANSWER_ENV["LLM_API_KEY"],
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": ADJUDICATE_PROMPT.format(pairs=listing)}],
            )
        try:
            labels = json.loads(response.choices[0].message.content)["labels"]
            verdict = {int(item["pair"]): bool(item["same"]) for item in labels}
        except (ValueError, KeyError, TypeError):
            verdict = {}
        for n, pair in enumerate(batch, 1):
            pair["same"] = verdict.get(n)

    batches = [pairs[i : i + BATCH] for i in range(0, len(pairs), BATCH)]
    await asyncio.gather(*(run(batch) for batch in batches))


def label(profile: str, convs: list[int]) -> None:
    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    all_pairs, report = [], {"per_conv": {}}
    relations = Counter()
    for conv in convs:
        data = json.loads(_dump_path(profile, conv).read_text())
        entities = data["entities"]
        relations.update(data["relations"])
        texts = [f"{e['name']}: {e['description']}" for e in entities]
        pairs = candidate_pairs(entities, list(embedder.embed(texts)))
        asyncio.run(adjudicate(pairs))
        for pair in pairs:
            pair["conv"] = conv
        confirmed = [p for p in pairs if p["same"]]
        dup_entities = {p["a"]["id"] for p in confirmed} | {p["b"]["id"] for p in confirmed}
        report["per_conv"][conv] = {
            "entities": len(entities),
            "candidate_pairs": len(pairs),
            "confirmed_pairs": len(confirmed),
            "entities_with_a_duplicate": len(dup_entities),
            "unlabelled": sum(p["same"] is None for p in pairs),
        }
        all_pairs.extend(pairs)

    labelled = [p for p in all_pairs if p["same"] is not None]
    signal_precision = {}
    for signal in ("containment", "token_jaccard", "embedding"):
        hits = [p for p in labelled if signal in p["signals"]]
        signal_precision[signal] = {
            "flagged": len(hits),
            "precision": round(sum(p["same"] for p in hits) / len(hits), 3) if hits else None,
        }
    total_entities = sum(c["entities"] for c in report["per_conv"].values())
    total_dup = sum(c["entities_with_a_duplicate"] for c in report["per_conv"].values())
    report.update(
        profile=profile,
        convs=convs,
        git={"sha": _git("rev-parse", "HEAD")},
        labeller="claude haiku via enhanced.llm_proxy (names + descriptions only)",
        note="Labels are an LLM judgment, not ground truth; recall is bounded by the blocking.",
        entities=total_entities,
        entities_with_a_duplicate=total_dup,
        duplicate_entity_fraction=round(total_dup / total_entities, 4) if total_entities else None,
        signal_precision=signal_precision,
        relation_names=dict(relations.most_common()),
    )
    out_dir = RESULTS_DIR / "entity-resolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"baseline-{profile}.json").write_text(json.dumps(report, indent=1))
    with open(out_dir / f"pairs-{profile}.jsonl", "w") as handle:
        handle.writelines(json.dumps(pair) + "\n" for pair in all_pairs)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "entities",
                    "entities_with_a_duplicate",
                    "duplicate_entity_fraction",
                    "signal_precision",
                )
            },
            indent=1,
        )
    )
    print(json.dumps(report["per_conv"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["dump", "label"])
    parser.add_argument("--profile", default="haiku")
    parser.add_argument("--convs", default="5-9")
    args = parser.parse_args()
    convs = _convs(args.convs)
    dump(args.profile, convs)
    if args.command == "label":
        label(args.profile, convs)


if __name__ == "__main__":
    main()
