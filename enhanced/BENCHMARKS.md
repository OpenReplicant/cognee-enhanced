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
