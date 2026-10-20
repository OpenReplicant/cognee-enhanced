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
