# cognee-enhanced

A fork of [topoteretes/cognee](https://github.com/topoteretes/cognee) (based on
v1.6.5, full upstream history kept so changes can be rebased and upstreamed)
that adds memory/retrieval enhancements **only when a benchmark shows they help**.

Everything specific to the fork lives here in `enhanced/`; enhancement code goes
into `cognee/` where upstream would put it, behind a flag or a retriever kwarg
that defaults to upstream behaviour until the hypothesis is accepted.

- [`HYPOTHESES.md`](HYPOTHESES.md): the register. Each entry states the claim,
  the mechanism, the metric, the prediction, and its status.
- [`BENCHMARKS.md`](BENCHMARKS.md): the ledger. One entry per accepted or
  rejected result, with the command that reproduces it.
- `bench/`: the harness. `results/`: committed result files.

## Workflow

1. **Register.** Add or refine the hypothesis in `HYPOTHESES.md` with its
   prediction and acceptance threshold, and commit it *before* running the
   treatment. A threshold chosen after seeing the numbers doesn't count.
2. **Baseline.** Run the baseline variant from the same commit as the
   treatment. Never compare against numbers from another commit or profile.
3. **Implement behind a switch.** Add the treatment as a variant in
   `bench/variants.py` (retriever kwargs or env flags). Upstream behaviour stays
   the default.
4. **Benchmark.** Commit the code, then run on the clean tree (the result
   file records the SHA and a `dirty` flag):
   ```bash
   uv run python -m enhanced.bench.run --name H1-lexical-lane --variants hybrid,hybrid_bm25
   uv run python -m enhanced.bench.compare \
     enhanced/results/locomo/H1-lexical-lane.json:hybrid \
     enhanced/results/locomo/H1-lexical-lane.json:hybrid_bm25
   ```
5. **Decide and record.** Commit the result JSON plus a `BENCHMARKS.md` entry
   in the same commit, and update the hypothesis status. Accepted: flip the
   default in a separate commit that cites the result. Rejected: keep the
   negative result and remove or park the code.

### Acceptance rule

- The primary metric's paired-bootstrap 95% CI lower bound is above 0 on the
  **test split**.
- No category shows a significant regression on the primary metric (CI upper
  bound below 0).
- p95 retrieval latency stays within the budget stated in the hypothesis.
- Anything tuned (weights, thresholds, routers) is tuned on the **dev split**
  only. For LoCoMo: dev = conversations 0-4, test = 5-9. Untuned changes can
  report all ten.

## Benchmark tiers

| Tier | What | LLM? | Status |
|---|---|---|---|
| **T0** | LoCoMo retrieval: evidence recall@k / all@k / MRR per question category, keyless profile (GLiNER graph + `bge-small` embeddings) | no | ready |
| T0b | LongMemEval-S session recall; MuSiQue / 2Wiki supporting-paragraph recall (multi-hop) | no | planned |
| M | Micro-benchmarks for one component (entity-resolution P/R, injection leakage, token cost) | varies | per hypothesis |
| T1r | T0's retrieval metrics on an **LLM-built** store (`haiku` profile) | ingest only | ready |
| T1 | QA accuracy (answer + LLM judge) on a fixed subset, cached responses | yes | planned (Haiku via the proxy) |
| T2 | Upstream full LoCoMo / BEAM evals (`cognee/eval_framework/`) | yes, paid | out of budget |

T0 is cheap, deterministic, and measures what most retrieval hypotheses change.
Its known blind spot: the keyless graph comes from the GLiNER demo extractor,
which is much weaker than an LLM-built graph. Graph-side hypotheses (H6, H7)
that pass T0 still need a T1 check before their default flips.

### LLM access: the Claude Code proxy

The OpenRouter key has a $0 limit, so LLM work goes through
`enhanced/llm_proxy.py`. It is an OpenAI-compatible endpoint that runs each
request as a headless `claude -p --model haiku` call (no tools, settings, MCP
or session persistence) and caches every response on disk by request hash:

```bash
uv run python -m enhanced.llm_proxy --port 8765 --model haiku &   # keep running
uv run python -m enhanced.bench.run --name ... --profile haiku --variants ...
```

The `haiku` profile points cognee's `custom` provider at the proxy
(`json_mode` instructor output) and keeps the keyless profile's local
embeddings, so the two profiles differ only in who builds the graph and
summaries. Pilot (10 windows): about 2 calls per document, about 9k
tokens per window, so a full LoCoMo store is about 10M tokens (about $2.60 at
Haiku list price) and about 2 hours at concurrency 4. Usage is logged to
`.cache/llm_proxy/haiku/usage.jsonl`. These calls count against the Claude
plan the session runs under.

### How T0 works

`bench/locomo_worker.py` ingests each conversation the way upstream's LoCoMo
eval does (an overview plus one document per 6-turn window, each window one
chunk), but without `improve()`. For each question it maps the evidence
`dia_id`s to gold windows, then records the windows a variant retrieves, in
rank order, to depth 20. A store is built once per (profile, conversation) under
`enhanced/.cache/` (gitignored) and reused by every variant. Keyless ingestion
takes roughly 20s per window on 4 CPU cores, so ingesting all ten
conversations is a one-time job of a few hours.

### Environment notes

- `bench/profiles.py` strips host `AWS_*` keys (they make ingestion use S3)
  and any LLM key, so the keyless profile stays keyless.
- Each ingest and retrieval run is its own subprocess, because cognee caches
  its config at import.
- Unit tests for the harness: `uv run pytest enhanced/tests -q`.
