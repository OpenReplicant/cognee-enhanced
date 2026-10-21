# Benchmark ledger

Newest first. Each entry: date, hypothesis, commit, result file, the compare
command, the numbers, and the decision. Numbers from different profiles or
commits are never compared with each other.

<!-- entries below -->

## 2026-10-20 · exploratory · H1 / summary-lane ablation · conv 0 · keyless

Not a decision. Conv 0 only (dev split), run while ingest of other
conversations shared the CPU, so latencies are inflated. Code at `b5f10041c`.
Files: `results/locomo/exploratory/H1-conv0.json`,
`results/locomo/exploratory/summary-ablation-conv0.json`.

| variant | R@5 | R@10 | MRR |
|---|---:|---:|---:|
| hybrid (upstream default) | 0.676 | 0.757 | 0.554 |
| hybrid_bm25 (H1) | 0.716 | 0.820 | 0.663 |
| hybrid_nosum | 0.745 | 0.832 | 0.614 |
| chunks (dense only) | 0.745 | 0.832 | 0.614 |
| bm25 only | 0.776 | 0.816 | 0.724 |
| hybrid_bm25_nosum | **0.806** | **0.850** | 0.714 |

n = 196 questions with evidence. hybrid_bm25 vs hybrid: R@5 +0.040
[−0.001, +0.083], R@10 +0.063 [+0.028, +0.102], MRR +0.109 [+0.068, +0.150].

```bash
uv run python -m enhanced.bench.compare \
  enhanced/results/locomo/exploratory/H1-conv0.json:hybrid \
  enhanced/results/locomo/exploratory/H1-conv0.json:hybrid_bm25
```

Takeaway: under keyless summaries, the summary lane lowers hybrid below
dense chunks alone. This led to registering H1b and to H2's split
prediction. Decisions use conversations 5-9.

## 2026-10-20 · exploratory · H3 cross-encoder rerank · conv 0 · keyless

Not a decision. Conv 0 (dev split), code at `cc9c272a7`; two ingests were
saturating the 4 cores, so latency here is an upper bound, not a measurement.
The bge-reranker-base run was stopped unfinished. File:
`results/locomo/exploratory/H3-conv0.json`.

| variant | R@5 | R@10 | MRR | p50 / p95 latency |
|---|---:|---:|---:|---:|
| hybrid_bm25 (H1) | 0.716 | 0.820 | 0.663 | 0.27 / 0.42 s |
| hybrid_bm25_pool30 (control) | 0.756 | 0.797 | 0.667 | 0.30 / 0.55 s |
| hybrid_bm25_rr_minilm | **0.866** | **0.915** | **0.786** | 6.5 / 9.9 s |

MiniLM vs control: R@5 +0.110 [+0.054, +0.168], MRR +0.120 [+0.070, +0.170],
37 wins / 12 losses (R@5). Quality is far above the registered prediction.
Latency on a contended CPU is far over the 400 ms budget. Next: measure on an
idle machine before the test-split decision.

## 2026-10-20 · H1 accepted (haiku, retrieval) · convs 5-9

Profile `haiku` (graph and summaries by Claude Haiku via the proxy, bge-small
embeddings). Test split, 982 questions with evidence. Code at `d007bec4a`
(clean). File: `results/locomo/H1-test-haiku.json`. The keyless ingest ran on
the same CPU throughout; every variant ran under the same load.

| variant | R@5 | R@10 | all@10 | MRR | p95 |
|---|---:|---:|---:|---:|---:|
| hybrid (upstream default) | 0.666 | 0.761 | 0.714 | 0.577 | 0.317 s |
| **hybrid_bm25 (H1)** | **0.744** | **0.830** | **0.782** | **0.652** | 0.351 s |
| hybrid_nosum | 0.620 | 0.708 | 0.662 | 0.524 | 0.238 s |
| hybrid_bm25_nosum | 0.735 | 0.816 | 0.766 | 0.636 | 0.277 s |
| chunks (dense only) | 0.620 | 0.708 | 0.662 | 0.524 | 0.107 s |
| bm25 only | 0.783 | 0.834 | 0.785 | 0.717 | 0.042 s |

H1 vs hybrid: R@5 +0.078 [+0.058, +0.099], R@10 +0.069 [+0.051, +0.087],
MRR +0.075 [+0.059, +0.092]. Per category R@5: adversarial +0.110,
single_hop +0.096, temporal +0.052, multi_hop +0.029 (n.s.), open_domain
+0.012 (n.s.). No significant regression anywhere.

```bash
uv run python -m enhanced.bench.compare \
  enhanced/results/locomo/H1-test-haiku.json:hybrid \
  enhanced/results/locomo/H1-test-haiku.json:hybrid_bm25
```

**Decision:** accepted on retrieval. The default stays off until (1) the BM25
index is cached instead of rebuilt from every chunk on each query, and (2) the
keyless test split agrees. Side findings: LLM summaries help (H2), and BM25
alone beats every equal-weight fusion (H2 lane weights).

## 2026-10-20 · H1 answer accuracy (T1) · haiku · convs 5-9

150 questions (30 per test conversation, seed 0; open_domain and upstream's
audited-corrupt golds left out). Haiku answers through each variant's own
completion path (upstream `unified.txt`) and judges with upstream's LoCoMo judge
prompt, via the proxy. Code at `94ee8d1cb`; the hybrid_bm25 workers for convs
5-8 ran the pre-cache lexical lane and conv 9 the cached one (rankings verified
identical on conv 5). File: `results/locomo-qa/H1-qa-haiku.json`.

| category | n | hybrid | hybrid_bm25 |
|---|---:|---:|---:|
| adversarial | 29 | 0.207 | 0.379 |
| temporal | 30 | 0.533 | 0.600 |
| multi_hop | 23 | 0.826 | 0.826 |
| single_hop | 68 | 0.838 | 0.779 |
| **all** | 150 | **0.653** | **0.673** |

Delta +0.020 [−0.047, +0.087]; 14 wins, 11 losses. Absolute accuracy is not
comparable with upstream's 0.907 (Haiku answering and judging, bge-small
embeddings, 20-chunk context).

**Decision:** H1's retrieval gain is not shown to reach answers. Most
single-hop losses keep the gold window at rank 1 but add same-person
distractor windows, so the default stays off until a precision fix is tested
on answers.

## 2026-10-20 · H3 rerank, quality accepted (haiku) · convs 5-9

Profile `haiku`, test split, 982 questions. Code at `6e164d0f0` (clean).
File: `results/locomo/H3-test-haiku.json`. CPU shared with the haiku dev ingest
and the entity dump, so latency is an upper bound.

| variant | R@5 | R@10 | MRR | p50 / p95 |
|---|---:|---:|---:|---:|
| hybrid (upstream, from H1 file) | 0.666 | 0.761 | 0.577 | 0.22 / 0.32 s |
| hybrid_bm25 (H1, from H1 file) | 0.744 | 0.830 | 0.652 | 0.25 / 0.35 s |
| hybrid_bm25_pool30 (control) | 0.759 | 0.831 | 0.655 | 0.24 / 0.41 s |
| **hybrid_bm25_rr_minilm** | **0.846** | **0.888** | **0.768** | 3.12 / 5.63 s |

MiniLM vs control: R@5 +0.086 [+0.065, +0.108], MRR +0.113 [+0.090, +0.135];
per category R@5: adversarial +0.103, temporal +0.097, single_hop +0.078,
multi_hop +0.068, open_domain +0.104 (n.s.).

```bash
uv run python -m enhanced.bench.compare \
  enhanced/results/locomo/H3-test-haiku.json:hybrid_bm25_pool30 \
  enhanced/results/locomo/H3-test-haiku.json:hybrid_bm25_rr_minilm
```

**Decision:** quality criteria met by a wide margin. The latency criterion
(≤ +400 ms p95, idle 4-core CPU) is not yet measured fairly, so the default
stays off until it is; over budget there means park, as registered.

## 2026-10-20 · H9 baseline: entity duplication in haiku stores · convs 5-9

Measurement, not a decision. Code at `57bfc4ed9` (`bench/entity_dups.py`).
Files: `results/entity-resolution/baseline-haiku.json`,
`results/entity-resolution/pairs-haiku.jsonl` (1,514 labelled candidate pairs).
Labels are Claude Haiku judging names + descriptions in batches of 25.

| conv | entities | candidate pairs | confirmed | entities with a duplicate |
|---|---:|---:|---:|---:|
| 5 | 420 | 372 | 82 | 132 |
| 6 | 527 | 268 | 66 | 100 |
| 7 | 485 | 324 | 88 | 114 |
| 8 | 417 | 260 | 31 | 56 |
| 9 | 393 | 290 | 86 | 123 |
| **all** | **2,242** | **1,514** | **353** | **525 (23.4%)** |

Excluding session-header entities: 811 pairs, 146 confirmed, 240 entities
(10.7%) with a duplicate. Blocking precision: containment 0.256, token
Jaccard 0.232, embedding (same-type mutual nearest, cos ≥ 0.93) 0.462 on 39
pairs.

## 2026-10-20 · H2 rejected (haiku) · convs 5-9

Dev tuning at `20a1f98f3` (`results/locomo/H2-tune-haiku.json`); test run at
`ddbdd4797` (clean), file `results/locomo/H2-test-haiku.json`.

| variant | R@5 | R@10 | MRR | R@1k tok | p95 |
|---|---:|---:|---:|---:|---:|
| hybrid_bm25 (H1) | 0.744 | 0.830 | 0.652 | 0.689 | 0.29 s |
| hybrid_bm25_w_haiku (0, 0.25, 1) | 0.778 | 0.862 | 0.712 | 0.738 | 0.26 s |

R@5 +0.034 [+0.016, +0.053], MRR +0.060 [+0.042, +0.078]. By category R@5:
single_hop +0.057, adversarial +0.050, temporal +0.029 (n.s.), open_domain
+0.004 (n.s.), **multi_hop −0.044 [−0.086, −0.002]**.

**Decision:** rejected under the no-significant-regression rule. The dev
data showed the same multi_hop drop (−0.033), so the follow-up H2b constrains
the tuning objective per category. That is a second, adaptive use of this
test split.

## 2026-10-20 · H2b rejected (haiku) · convs 5-9 · second use of the test split

Constrained dev tuning at `fe5377f24`
(`results/locomo/H2-tune-haiku-maxdrop0.01.json`: best eligible (0.5, 0.25, 1),
dev R@5 0.7927, worst category −0.006). Test run at `ebb007154` (clean), file
`results/locomo/H2b-test-haiku.json`.

| variant | R@5 | R@10 | MRR | R@1k tok |
|---|---:|---:|---:|---:|
| hybrid_bm25 (H1) | 0.744 | 0.830 | 0.652 | 0.689 |
| hybrid_bm25_w2_haiku (0.5, 0.25, 1) | 0.749 | 0.827 | 0.663 | 0.709 |

R@5 +0.005 [−0.007, +0.018]; multi_hop −0.038 [−0.072, −0.003].

**Decision:** rejected. Lane-weight tuning is closed on this data. Equal
weights stay.

## 2026-10-20 · H3 answer accuracy (T1), inconclusive · plus a determinism bug

Same 150 questions as the H1 QA run. Code at `b4eb3b024`. File:
`results/locomo-qa/H3-qa-haiku.json`.

| category | n | hybrid | hybrid_bm25 | hybrid_bm25_rr_minilm |
|---|---:|---:|---:|---:|
| adversarial | 29 | 0.310 | 0.414 | 0.345 |
| multi_hop | 23 | 0.739 | 0.783 | 0.609 |
| single_hop | 68 | 0.868 | 0.897 | 0.853 |
| temporal | 30 | 0.667 | 0.700 | 0.700 |
| **all** | 150 | **0.700** | **0.747** | **0.687** |

Rerank vs H1: −0.060 [−0.120, 0.000], 6 wins / 15 losses.

**Reproducibility check failed:** hybrid and hybrid_bm25 had already been
answered in the H1 QA run. On rerun only 48/150 (hybrid) and 45/150
(hybrid_bm25) answers were identical, and 31/29 verdicts flipped; accuracy
moved 0.653 → 0.700 and 0.673 → 0.747 with no code change on their path.
Cause: the hybrid context's entity-edge bullets came out in a
process-dependent order, and sometimes a different set made the cut, so
prompts differed and the proxy cache missed (704 of 900 calls). Fixed in
`5b9cd17c9`; contexts are now byte-identical across processes.

**Decision:** neither QA run supports a conclusion. Rerun on the fixed code
with 60 questions per conversation, then repeat once to confirm it replays
from cache.

## 2026-10-20 · QA on deterministic contexts (T1) · haiku · convs 5-9 · n=300

60 questions per test conversation (seed 0), deterministic hybrid context
(fix `5b9cd17c9`). The run started at `9178cc327` and finished at `7145361c6`;
the result reads `dirty` because the uncommitted PPR/graph-cache refactor was
on disk for later workers (lexical-lane behaviour unchanged by tests; a
cache-replay repeat run is the end-to-end check). File:
`results/locomo-qa/QA-det-haiku.json`.

| category | hybrid | hybrid_bm25 | hybrid_bm25_rr_minilm |
|---|---:|---:|---:|
| adversarial | 0.296 | 0.310 | 0.352 |
| multi_hop | 0.744 | 0.767 | 0.698 |
| single_hop | 0.892 | 0.854 | 0.908 |
| temporal | 0.625 | 0.625 | 0.661 |
| **all** | **0.680** | **0.670** | **0.700** |

hybrid_bm25 vs hybrid −0.010 [−0.060, +0.037]; rerank vs hybrid +0.020
[−0.030, +0.070]; rerank vs hybrid_bm25 +0.030 [−0.020, +0.080]. Nothing
significant.

**Reading:** with 20 chunks (~6.5k passage tokens) in the prompt, answer
accuracy hardly responds to ranking (+18 pt recall@5 here moves answers by
+2 pt n.s.). The evidence is almost always in the 20 and the reader finds it.
Next: QA at a small context (`chunks_top_k=5`), where ranking should
matter. That is also the first H4 test.

**Reproducibility check (repeat at `681d06481`, clean):**
`results/locomo-qa/QA-det-haiku-repeat.json` replayed every answer and
verdict: 900/900 answers, context sizes and verdicts identical across the
three variants, including the lexical lane after the graph-cache refactor.
T1 results on the fixed code reproduce exactly, so the `dirty` flag on the
first run is resolved.

## 2026-10-21 · H6 PPR lane rejected on LoCoMo (haiku) · convs 5-9

Code at `83c19e339` (clean). File: `results/locomo/H6-test-haiku.json`.

| variant | R@5 | all@10 | MRR | multi_hop all@10 | p95 |
|---|---:|---:|---:|---:|---:|
| hybrid | 0.666 | 0.714 | 0.577 | 0.314 | 0.32 s |
| hybrid_ppr | 0.614 | 0.720 | 0.509 | 0.329 | 0.30 s |
| hybrid_bm25 | 0.744 | 0.782 | 0.652 | 0.364 | 0.32 s |
| hybrid_bm25_ppr | 0.717 | 0.785 | 0.592 | 0.386 | 0.34 s |

hybrid_bm25_ppr vs hybrid_bm25: multi_hop all@10 +0.021 [−0.021, +0.071];
recall@5 −0.027 [−0.047, −0.007]; MRR −0.060 [−0.079, −0.040].

**Decision:** rejected on LoCoMo (primary effect n.s., guard violated). The
likely cause is speaker-entity hubs; H6 stays open for document multi-hop QA.

## 2026-10-21 · H4a QA at five passages (T1) · haiku · n=300

Same 300 questions as `QA-det-haiku`, deterministic contexts. Code at
`2e890c720` (clean). File: `results/locomo-qa/H4a-qa-haiku.json`.

| variant | passages | adversarial | multi_hop | single_hop | temporal | **all** | context chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 20 | 0.296 | 0.744 | 0.892 | 0.625 | **0.680** | 29.4k |
| hybrid_bm25 | 20 | 0.310 | 0.767 | 0.854 | 0.625 | 0.670 | 29.7k |
| hybrid_bm25_rr_minilm | 20 | 0.352 | 0.698 | 0.908 | 0.661 | 0.700 | 30.5k |
| hybrid_k5 | 5 | 0.338 | 0.721 | 0.862 | 0.625 | 0.673 | 15.2k |
| hybrid_bm25_k5 | 5 | 0.380 | 0.721 | 0.900 | 0.607 | 0.697 | 15.3k |
| hybrid_bm25_rr_minilm_k5 | 5 | 0.437 | 0.744 | 0.877 | 0.607 | 0.703 | 15.6k |

rerank_k5 vs hybrid_k5 +0.030 [−0.023, +0.083]; rerank_k5 vs hybrid@20
+0.023 [−0.033, +0.080]; hybrid_k5 vs hybrid@20 −0.007 [−0.057, +0.040].

**Decision:** prediction (1) not met, (2) met. No default change from this
alone. The flat 0.67-0.70 band points at the reader as the limit; next is
error attribution.

## 2026-10-21 · Error attribution: most wrong answers had the evidence in context

`bench/attribute.py` joins a QA run with the retrieval run of the same
retriever (deterministic chunk lane) and asks, for each wrong answer, whether
the gold windows were in the passages the reader saw. Entity and fact lines
are not credited, so "no gold shown" is an upper bound on retrieval misses.

| run | wrong | no gold shown | some gold shown | **all gold shown** |
|---|---:|---:|---:|---:|
| hybrid, k=20 (`QA-det-haiku`) | 96 | 14 | 8 | **74** |
| hybrid, k=5 (`H4a-qa-haiku`) | 98 | 32 | 12 | **54** |
| H1 + rerank, k=20 (`QA-det-haiku`) | 90 | 5 | 10 | **75** |

By category (hybrid, k=20), wrong answers with all gold shown: adversarial
46/50, temporal 12/21, single_hop 10/14, multi_hop 6/11 (5 more partial).

```bash
uv run python -m enhanced.bench.attribute \
  enhanced/results/locomo-qa/QA-det-haiku.json:hybrid \
  enhanced/results/locomo/H1-test-haiku.json:hybrid --k 20
```

**Reading:** with Haiku as the reader, about three quarters of LoCoMo errors
happen after retrieval succeeded, and adversarial questions dominate (the
reader answers about the wrong speaker instead of declining; upstream's
gpt-5-mini reader scored 0.892 there with the same prompt). Retrieval gains
can move at most ~15-20 of 300 answers here. They matter more where the
evidence cannot all fit in the context (larger stores, BEAM-10M-like scale),
and reader-side fixes (H14 abstention, an attribution check) target the
larger error mass.

## 2026-10-21 · H3 parked: rerank latency over budget on an idle machine

Code at `950c2ce5a` (clean), haiku test split, 982 queries, CPU idle except
the run itself (load ~3.8 = the reranker on 4 cores). File:
`results/locomo/H3-latency-idle.json`.

| variant | R@5 | MRR | p50 | p95 |
|---|---:|---:|---:|---:|
| hybrid_bm25_pool30 (control) | 0.759 | 0.655 | 0.21 s | 0.25 s |
| hybrid_bm25_rr_minilm | 0.846 | 0.768 | 1.55 s | 2.16 s |

Rerank rankings reproduced the earlier run exactly (987/987). p95 +1.91 s
against a +0.40 s budget, so parked as registered; H3b tests a 12-chunk,
800-char rerank.

## 2026-10-21 · T0b-1: H1 / H3 on LongMemEval-S (fresh data) · chunks-only

70 questions (10 per type incl. abstention, seed 0), one chunks-only store per
question (~250 dated rounds, bge-small, no LLM: hybrid = dense chunks here).
Session-level gold. Code at `f7d7cb702` (clean). File:
`results/longmemeval/T0b-1.json`.

| variant | R@1 | R@5 | all@5 | MRR | p95 |
|---|---:|---:|---:|---:|---:|
| hybrid (dense chunks) | 0.564 | 0.864 | 0.814 | 0.880 | (cold start) |
| hybrid_bm25 (H1) | 0.556 | 0.886 | 0.829 | 0.895 | 0.30 s |
| bm25 | 0.555 | 0.837 | 0.771 | 0.853 | 0.07 s |
| hybrid_bm25_pool30 | 0.556 | 0.886 | 0.829 | 0.894 | 0.28 s |
| hybrid_bm25_rr_minilm (H3) | **0.642** | **0.934** | **0.886** | **0.957** | 2.27 s |

H1: R@5 +0.021 [−0.021, +0.071], not replicated. H3: MRR +0.063 [+0.024,
+0.108], R@5 +0.049 [+0.013, +0.094], replicated.

## 2026-10-21 · H3b rejected: truncated rerank loses evidence

Haiku test split, idle CPU, code at `048be8789` (clean). File:
`results/locomo/H3b-test-haiku.json`.

| variant | R@5 | R@10 | MRR | p50 / p95 |
|---|---:|---:|---:|---:|
| hybrid_bm25 (H1) | 0.744 | 0.830 | 0.652 | 0.17 / 0.21 s |
| hybrid_bm25_rr12_800 | 0.712 | 0.779 | 0.677 | 0.36 / 0.46 s |

MRR +0.025 [+0.003, +0.048]; R@5 −0.032 [−0.055, −0.008]; R@10 −0.050
[−0.067, −0.034]. Latency within budget; quality regresses, so rejected.
Prefix truncation drops the evidence turn in long windows.

## 2026-10-21 · H1b accepted, H1 accepted on keyless · convs 5-9

Keyless profile (GLiNER graph, bge-small). Code at `2cb780466` (clean). File:
`results/locomo/H1-test-keyless.json`.

| variant | R@5 | R@10 | MRR | p95 |
|---|---:|---:|---:|---:|
| hybrid (upstream) | 0.573 | 0.666 | 0.483 | 0.18 s |
| hybrid_bm25 (H1) | 0.648 | 0.779 | 0.576 | 0.20 s |
| hybrid_nosum | 0.620 | 0.708 | 0.524 | 0.12 s |
| **hybrid_bm25_nosum (H1b)** | **0.735** | **0.816** | **0.640** | 0.15 s |
| chunks (upstream keyless recall) | 0.620 | 0.708 | 0.524 | 0.06 s |
| bm25 | 0.783 | 0.834 | 0.717 | 0.03 s |

H1b vs chunks: R@5 +0.116 [+0.092, +0.140], MRR +0.116 [+0.097, +0.135].
H1 vs hybrid: R@5 +0.076 [+0.057, +0.096]. No category regresses in either.

**Decision:** H1b accepted. Keyless `recall()` (upstream: dense `CHUNKS`)
is the place to use it; flip proposed as this fork's default.
