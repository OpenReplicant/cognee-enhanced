# Hypothesis register

Every enhancement starts here. An entry is a falsifiable claim with a primary
metric, a prediction set before the treatment runs, and the tier that tests it
(tiers and the acceptance rule are in [README.md](README.md)). Paper figures are
as reported by their authors; we haven't reproduced them.

**Status:** `proposed` → `registered` (prediction committed) → `running` →
`accepted` / `rejected` / `parked`.

## What upstream v1.6.5 already does

The earlier idea list (written against the v1.0 docs) proposed several things
the code already does. These are **not** hypotheses here; we build on them.

| Idea | Already upstream as |
|---|---|
| Hierarchical summary tree (RAPTOR) / global sensemaking | `global_context_index` improve stage + the global-context lane in `HybridRetriever` |
| Hybrid chunk + graph retrieval by default | `HYBRID_COMPLETION` default: RRF over dense chunk and dense summary ranks, plus an entity/fact lane |
| Write-time conflict handling (partly) | `improve(review_conflicts=True)`: `FactConflict` nodes, per-fact status, date-ordered values (opt-in, post-hoc, LLM) |
| Time-aware retrieval | `Timestamp` nodes from the default extraction prompt + `TEMPORAL` retriever |
| Reflection / distillation | `distill_sessions`, `update_user_preferences`, `build_truth_subspace` stages |
| Procedural memory | `Skill`, `SkillRun`, `CandidateSkill`, `SkillImprovementProposal` + `SKILLS` search |
| Query decomposition / agentic loop | `GRAPH_COMPLETION_DECOMPOSITION`, `AGENTIC_COMPLETION` |
| Incremental indexing | `update()`: chunk-level diff and re-extraction |
| Citations | `include_references` on hybrid |
| Published memory benchmarks | LoCoMo 0.907 (gpt-5-mini answers, gpt-5.1 judge), BEAM 100K 0.79 and 10M 0.67 (`cognee/eval_framework/`) |

Where upstream's own results are weakest: BEAM at 10M scale (0.67; the
report says no single retrieval setting suited every question type), BEAM
multi-session and temporal reasoning (0.50 each at 100K, on 2 questions per
ability), and LoCoMo open-domain (0.59). The hypotheses target those, plus the
retrieval mechanisms the code lacks entirely: a lexical lane in hybrid, a
reranker, PPR, entity resolution, token budgeting and learned routing.

Gaps confirmed in the code (v1.6.5):
- `hybrid/ranking.py` fuses only dense-chunk and dense-summary ranks. BM25
  exists (`CHUNKS_LEXICAL`) but never feeds hybrid.
- Entity names are only normalised (`generate_node_name`: lower-case, strip
  `'`). There is no alias or embedding-based merge.
- No cross-encoder reranking anywhere; "rerank" in the code means the temporal
  stable partition.
- PageRank is used only for the hub list in `GRAPH_REPORT`, not for retrieval.
- `recall()` routing is a rule table (quoted phrase, coding rules, absolute
  date, else hybrid).
- `forget()` deletes by data item or dataset; there's no fact-level retraction.
- No token budget on retrieved context.

---

## Tier T0: retrieval, LLM-free (start here)

### H1: A lexical lane in hybrid RRF raises chunk recall
- **Status:** registered
- **Claim:** Adding the BM25 rank as a third input to the hybrid chunk lane's
  RRF (beside the dense-chunk and dense-summary ranks) raises recall@5.
- **Why:** Conversational memory questions name people, objects and places
  ("Caroline's painting", "the Mumford & Sons concert") that dense
  bi-encoders blur. Hybrid sparse+dense fusion beats either alone on BEIR-style
  evaluations, and "RAG vs. GraphRAG" (Han et al. 2025) finds no single
  retrieval view wins on every question type.
- **Metric:** LoCoMo recall@5 (primary), MRR, per category; p95 latency.
- **Prediction:** +2 pt recall@5 overall; largest on single_hop; no category
  regresses significantly; p95 latency +≤50 ms.
- **Change:** `hybrid_lexical_lane=True` kwarg on `HybridRetriever`, reusing
  `BM25ChunksRetriever` scoring.

### H2: Lane weights tuned per corpus beat equal-weight RRF
- **Status:** proposed (run after H1)
- **Claim:** Weighted RRF (per-lane weight; the RRF k is already adaptive)
  tuned on dev conversations generalises to test conversations.
- **Why:** The BEAM report found the best channel balance depends on the
  question mix. The keyless smoke run already ranks dense chunks alone above
  hybrid on MRR, which hints that the summary lane can add noise when the
  summaries are weak.
- **Metric:** recall@5 on test split (conversations 5-9).
- **Prediction:** +1 pt over H1's equal weights; the summary-lane weight
  falls below 1 under the keyless profile.

### H3: Cross-encoder reranking of the fused top-N raises MRR
- **Status:** proposed
- **Claim:** Reranking the top 30 fused chunks with a small cross-encoder
  (fastembed `TextCrossEncoder`, e.g. `BAAI/bge-reranker-base`) raises MRR and
  recall@5.
- **Why:** Retrieve-then-rerank is the standard gain on MS MARCO and BEIR.
  Cognee has no reranker at all.
- **Metric:** MRR (primary), recall@5; p95 latency per query on CPU.
- **Prediction:** MRR +5 pt; recall@5 +3 pt; p95 latency ≤ +400 ms on 4 CPU
  cores. Rejected if latency exceeds that budget even when quality improves
  (record the result and park).

### H4: Token-budgeted packing beats a fixed top-k
- **Status:** proposed
- **Claim:** Selecting chunks to a token budget, cutting where the fused
  score falls off, gives higher evidence recall per context token than a
  fixed `chunks_top_k`.
- **Why:** Mem0 and Zep both lead with token and latency savings at equal
  accuracy. A fixed top-20 at 4096-token windows is a large prompt.
- **Metric:** evidence recall at fixed budgets of 1k / 2k / 4k tokens; mean
  tokens per query at matched recall@10. The harness needs a token-count field
  for this.
- **Prediction:** at matched recall, ≥25% fewer context tokens.

### H5: A learned query router beats fixed hybrid
- **Status:** proposed (needs ≥3 accepted or measured variants first)
- **Claim:** A light router (kNN over question embeddings → best variant,
  fitted on dev conversations' per-question outcomes) beats the best single
  variant on test conversations.
- **Why:** Adaptive-RAG (Jeong et al. 2024) matched multi-step accuracy at
  lower cost by routing on question complexity. Upstream's BEAM 10M result
  came from routing on *gold* question-type labels, and their report names
  an inferred router as the next step. `recall()` today routes by regex.
- **Metric:** test-split recall@5 vs. best single variant; router accuracy.
- **Prediction:** +1.5 pt recall@5; routing is a no-op for the ~70% of
  questions where variants tie.

### H6: A Personalized PageRank lane helps multi-hop
- **Status:** proposed
- **Claim:** A PPR lane (seeded by entities the query matches, run over the
  entity–chunk graph, ranking chunks by PPR mass) fused into hybrid raises
  `all@10` on multi-hop questions without hurting single-hop.
- **Why:** HippoRAG 2 (Gutiérrez et al. 2025) reports about +7 pt over the
  best dense retriever on associative multi-hop tasks, with no loss on factual
  QA. Cognee already stores both chunk and entity nodes, so the graph exists.
- **Metric:** all@10 on multi_hop (primary); recall@5 overall as a guard.
- **Prediction:** +4 pt all@10 multi_hop on LoCoMo; larger on MuSiQue/2Wiki
  (T0b). Under the keyless graph the effect may be small. Run T1 before
  rejecting.

### H7: Fact-augmented index keys help multi-hop and temporal
- **Status:** proposed
- **Claim:** Indexing each chunk under extra keys (its extracted facts as
  text, plus the session date) and fusing the extra key matches raises
  recall@5 for multi_hop and temporal.
- **Why:** LongMemEval (Wu et al. 2024) reports that fact-augmented keys
  and time-aware query expansion each raise recall and QA accuracy
  significantly.
- **Metric:** recall@5 by category.
- **Prediction:** +3 pt multi_hop, +2 pt temporal. Needs re-ingestion (a new
  store); keyless facts are weak, so a T1 run decides.

### H8: A recency prior helps knowledge-update questions (T0b)
- **Status:** proposed (needs LongMemEval-S in the harness)
- **Claim:** Multiplying fused scores by a decay on the source session date
  raises recall of the *latest* evidence for knowledge-update questions, and
  leaves other categories flat.
- **Why:** Generative Agents (Park et al. 2023) and MemoryBank use
  recency × importance × relevance. LongMemEval's knowledge-update category
  rewards the latest value. The hybrid ranking has importance, truth and
  personal factors but no recency factor.
- **Metric:** LongMemEval-S knowledge-update session recall@5.
- **Prediction:** +5 pt KU; no other category changes by more than 1 pt.

## Tier M: component micro-benchmarks

### H9: Entity resolution removes alias duplicates at high precision
- **Status:** proposed
- **Claim:** An entity-resolution stage (string and embedding blocking, then
  a merge decision, then canonical-ID rewrite) cuts duplicate entity nodes
  substantially without false merges.
- **Why:** Duplicate entities are the dominant LLM-KG construction error
  (the GraphRAG and LightRAG ablations both show it). Cognee normalises case
  only.
- **Metric:** pairwise merge precision and recall on a labelled alias set
  (LoCoMo speaker and entity aliases plus synthetic variants); downstream:
  hybrid entity-lane recall (T0) and H6 multi-hop.
- **Prediction:** precision ≥0.95 at recall ≥0.6 with no LLM in the merge
  decision; an LLM adjudicator for the uncertain band lifts recall further.

### H10: Quarantining instruction-like captures cuts injection leakage
- **Status:** proposed
- **Claim:** Tagging captured content with a source-trust level and
  stripping or quarantining instruction-like spans before graph promotion cuts
  the injected instructions that resurface in `recall()` context by ≥90%, at
  ≤1 pt recall@5 cost on clean data.
- **Why:** Auto-captured memory is a persistent prompt-injection channel.
  `agent_memory/sanitization.py` exists, so we measure what it already catches
  before adding anything.
- **Metric:** leakage rate on a seeded-injection corpus; LoCoMo recall@5
  guard.

## Tier T1/T2: need an LLM (blocked until the key has budget)

### H11: Write-time reconciliation beats post-hoc conflict review
- **Status:** proposed
- **Claim:** Mem0-style ADD/UPDATE/DELETE/NOOP decisions on write give
  higher knowledge-update and contradiction-resolution accuracy than
  `review_conflicts` at equal or lower LLM tokens per ingested turn.
- **Why:** Mem0 (Chhikara et al. 2025) reports about a 26% relative gain on
  LoCoMo's LLM-judge metric and large token/latency savings. MemoryAgentBench
  (Hu et al. 2025) found every system it tested weak on conflict resolution.
- **Metric:** LongMemEval-S KU accuracy; BEAM contradiction-resolution and
  knowledge-update rubric; tokens per turn.

### H12: Edge validity intervals help temporal and KU questions
- **Status:** proposed
- **Claim:** First-class `valid_at` / `invalid_at` on relationship edges (set
  at write time and by H11's invalidations), with as-of retrieval, raises
  temporal-reasoning and KU accuracy.
- **Why:** Zep/Graphiti (Rasmussen et al. 2025) reports up to about +18.5% on
  LongMemEval with lower latency, largely from bi-temporal edges. Cognee has
  `Timestamp` nodes and per-fact conflict status, but no edge validity
  intervals.
- **Metric:** LongMemEval-S temporal and KU accuracy; BEAM event ordering.

### H13: Sleep-time precomputation lowers query-time cost at equal accuracy
- **Status:** proposed
- **Claim:** An offline improve stage that anticipates likely questions and
  stores the derived answers or insights lets a cheaper query-time
  configuration match the default's accuracy.
- **Why:** Sleep-time compute (Lin et al. 2025) reports about 5× less
  test-time compute at equal accuracy on stateful reasoning tasks.
- **Metric:** accuracy vs. query-time tokens curve on LongMemEval-S.

### H14: Calibrated abstention
- **Status:** proposed
- **Claim:** Gating the answer on retrieval-score features ("not in memory"
  below a threshold tuned on dev) raises abstention F1 without lowering
  answerable accuracy.
- **Metric:** LoCoMo adversarial and LongMemEval abstention accuracy;
  answerable accuracy as a guard.

## Parked: integration-layer ideas (other repo)

These belong in `topoteretes/cognee-integrations`, not in this fork's core,
and need an agent-task benchmark we don't have yet: memory-in-the-loop core
block (MemGPT/Letta), unifying the framework wrappers on the v1 API, native
backends (LangGraph `BaseStore`, ADK `MemoryService`, OpenAI Agents
`Session`), multi-agent scoped memory, and the frameworks still missing.
