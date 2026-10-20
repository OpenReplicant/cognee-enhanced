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
- **Status:** accepted (haiku profile, retrieval). Default flip waits on a cached BM25 index (the lane rescans all chunks per query) and the keyless test split.
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
- **Change:** `HybridRetriever(lexical_lane=True)`, reusing
  `BM25ChunksRetriever` scoring (commit `b5f10041c`). Variant `hybrid_bm25`.
- **Exploration (conv 0, keyless; not a decision):** hybrid 0.676 → hybrid_bm25
  0.716 R@5 (+4.0 [−0.1, +8.3]); MRR +10.9 [+6.8, +15.0]. In the same run
  hybrid scored *below* both of its inputs alone (dense chunks 0.745, BM25
  0.776). See H2. File: `results/locomo/exploratory/H1-conv0.json`.
- **Decision split:** conversation 0 has been seen, so the decision uses the
  **test split, conversations 5-9**, with the prediction unchanged.
- **Result (haiku, convs 5-9, n=982):** R@5 0.666 → 0.744, **+7.8 pt
  [+5.8, +9.9]**; MRR +7.5 [+5.9, +9.2]; largest on adversarial (+11.0) and
  single_hop (+9.6); no category regresses (multi_hop MRR −0.1 [−5.7, +5.4]);
  p95 latency 0.317 → 0.351 s (+34 ms). Every criterion met. File:
  `results/locomo/H1-test-haiku.json`.
- **Answer accuracy (T1, haiku answers and judge, 150 test questions):**
  0.653 → 0.673, +2.0 pt [−4.7, +8.7], 14 wins / 11 losses, so not
  established. Mixed by category: adversarial 0.21 → 0.38, temporal
  0.53 → 0.60, multi_hop flat, single_hop 0.84 → 0.78. In 6 of 8 single-hop
  losses the gold window is rank 1 under *both* variants; the BM25 context adds
  keyword-matching windows about the same person's other activities, and the
  answerer picks one. Recall at depth 20 rose and precision of the context
  fell. Hence the default stays off until a precision fix (H3 rerank, H4
  budget) is tested on answers. File: `results/locomo-qa/H1-qa-haiku.json`.

### H1b: Keyless retrieval should fuse BM25 with dense chunks
- **Status:** registered
- **Claim:** For keyless installs (where `recall()` already falls back to
  `CHUNKS`), dense-chunk + BM25 RRF beats dense chunks alone.
- **Why:** The conv 0 exploration above. This needs no LLM, so it ships to
  every no-key user.
- **Metric:** recall@5 (primary), MRR; test split, conversations 5-9.
  `hybrid_bm25_nosum` vs `chunks`.
- **Prediction:** +3 pt recall@5, +5 pt MRR; no category regresses
  significantly. (Conv 0 showed +6.1 R@5, +10.0 MRR; the prediction is set
  lower on purpose because that was the conversation we looked at.)

### H2: Lane weights tuned per corpus beat equal-weight RRF
- **Status:** rejected (haiku): overall gain met, but multi_hop regressed significantly. Follow-up H2b.
- **Claim:** Weighted RRF (per-lane weight; the RRF k is already adaptive)
  tuned on dev conversations generalises to test conversations.
- **Why:** The BEAM report found the best channel balance depends on the
  question mix. On conv 0 (keyless), removing the dense summary lane raised
  hybrid from 0.676 to 0.745 R@5, the same as dense chunks alone, and dense +
  BM25 without summaries reached 0.806 R@5 / 0.714 MRR, the best of every
  variant tried. Keyless summaries are assembled from GLiNER extractions, so
  this may not hold for LLM-written summaries. That is the open question.
- **Procedure:** `bench/tune_lanes.py` dumps each dev question's chunk-lane
  candidates with their dense / summary / lexical ranks, then grid-searches
  weights (dense, summary) ∈ {0, .25, .5, .75, 1, 1.5, 2, 3}², lexical fixed
  at 1 (RRF order is scale-invariant), maximising recall@5 on **dev
  (conversations 0-4)** through the real `rank_chunk_summary_pairs`.
  Equal-weight offline fusion reproduced all 158 committed `hybrid_bm25`
  rankings on conv 5. The best dev weights then run through the real
  retriever (`HybridRetriever(lexical_lane=True, chunk_lane_weights=...)`) on
  the test split against `hybrid_bm25`, once per profile.
- **Metric:** recall@5 on test split (conversations 5-9).
- **Prediction (haiku):** +1 pt recall@5 over H1's equal weights, with the
  lexical weight above both dense weights (from the BM25-alone result).
- **Prediction (keyless):** summary-lane weight tuned on convs 0-4 ends below
  0.5, and `hybrid_bm25_nosum` beats `hybrid` by ≥5 pt R@5 on 5-9.
  **Prediction (llm, T1):** the tuned summary weight stays ≥0.5. If it does,
  lane weights must be profile-specific, not a single new default.
- **Evidence so far (haiku, convs 5-9; ablations, not the tuned test):**
  removing the summary lane costs −4.6 pt R@5 [−6.3, −3.0], so LLM summaries
  help, as predicted. BM25 *alone* (R@5 0.783, MRR 0.717) beats every
  equal-weight fusion, including H1 (0.744 / 0.652): the dense lanes dilute
  the lexical ranking, so a lexical weight > 1 is the obvious candidate.
  The embedder (bge-small) is weaker than upstream's default, which confounds
  this. Tuning needs haiku stores for convs 0-4.
- **Result (haiku).** Dev tuning (`H2-tune-haiku.json`): equal 0.7715 → best
  (dense 0, summary 0.25, lexical 1) 0.7948. Test split (`H2-test-haiku.json`,
  n=982): R@5 0.744 → 0.778, **+3.4 pt [+1.6, +5.3]**; MRR +6.0 [+4.2,
  +7.8]; single_hop +5.7, adversarial +5.0; but **multi_hop −4.35 pt [−8.6,
  −0.2]**, a significant regression, so rejected under the acceptance
  rule. The same drop was visible on dev (multi_hop −3.3) and the objective
  ignored it: zeroing the dense-chunk lane removes the semantic signal
  multi-hop questions need. (`hybrid_bm25` reproduced all 987 H1 rankings
  in this run.)

### H2b: Category-constrained lane weights
- **Status:** rejected (haiku). Lane-weight tuning closed on this data.
- **Claim:** Weights chosen on dev to maximise recall@5 *subject to no
  category losing more than 1 pt vs equal weights on dev* improve test
  recall@5 over H1 without a significant regression in any category.
- **Procedure:** `tune_lanes tune --max-category-drop 0.01` on dev convs 0-4
  (haiku), then the best eligible weights through the real retriever on
  convs 5-9 against `hybrid_bm25`.
- **Prediction:** +1.5 pt recall@5 [CI above 0]; multi_hop within ±1.5 pt.
- **Caveat:** this is the test split's second use for lane weights. H2's
  test result motivated the constraint, which is adaptive reuse; the ledger
  says so. A fresh check needs other data (LongMemEval / another profile).
- **Result (haiku, n=982, `H2b-test-haiku.json`):** dev-constrained weights
  (0.5, 0.25, 1). R@5 +0.5 pt [−0.7, +1.8] (prediction +1.5); MRR +1.1
  [+0.0, +2.1]; **multi_hop −3.8 pt [−7.2, −0.3]** again, though dev showed
  only −0.5. Rejected.
- **Conclusion for H2/H2b:** on this data, down-weighting the dense lanes
  reliably trades multi-hop for single-hop, and the 140-question multi-hop
  slice is too small for a dev constraint to transfer. More tuning here would
  fit the test split. Equal-weight RRF stays. The multi-hop-safe gains so far
  come from reranking (H3: multi_hop +6.8 R@5), which reorders by
  query-passage relevance instead of by lane.

### H3: Cross-encoder reranking of the fused top-N raises MRR
- **Status:** quality accepted (haiku, MiniLM); latency verdict pending an idle-machine measurement
- **Claim:** Fusing the chunk lane to a pool of 30, reranking that pool with a
  local cross-encoder, and cutting to 20 raises MRR and recall@5 over the same
  pipeline without the rerank.
- **Why:** Retrieve-then-rerank is the standard gain on MS MARCO and BEIR.
  Cognee has no reranker at all.
- **Change:** `HybridRetriever(rerank_top_n=30, rerank_model=...)` with
  fastembed `TextCrossEncoder` (`cognee/modules/retrieval/hybrid/rerank.py`);
  a model failure keeps the fused order.
- **Variants:** base `hybrid_bm25` (H1). Control `hybrid_bm25_pool30` (pool
  of 30, no rerank), which separates the larger pool's effect from the
  reranker's. Treatments `hybrid_bm25_rr_minilm` (ms-marco-MiniLM-L-6, 80 MB)
  and `hybrid_bm25_rr_bge` (bge-reranker-base, 1 GB). The decision compares
  each treatment against the **control**.
- **Metric:** MRR (primary), recall@5; p95 retrieval latency, measured on
  an idle machine for the decision run.
- **Prediction:** MRR +5 pt and recall@5 +3 pt for at least one model;
  p95 latency ≤ +400 ms on 4 CPU cores. Above that budget the result is
  recorded and the change parked, even if quality improves. Test split,
  conversations 5-9, both profiles.
- **Result (haiku, convs 5-9, n=982; MiniLM vs pool-30 control):** R@5
  0.759 → 0.846, **+8.6 pt [+6.5, +10.8]**; MRR 0.655 → 0.768, **+11.3 pt
  [+9.0, +13.5]**; every category improves (open_domain R@5 n.s., n=45).
  Against upstream hybrid: +18.0 R@5, +19.1 MRR. Latency p50 3.1 s / p95
  5.6 s vs 0.24 / 0.41 s, but measured while an ingest and a QA run shared
  the CPU. The registered latency check needs an idle machine; until then
  the change stays off. File: `results/locomo/H3-test-haiku.json`. bge
  untested on the test split.

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
- **Tooling:** `bench/budget.py` computes recall at 1k / 2k / 4k passage
  tokens offline from any committed result (cl100k counts from the stores).
  Baselines on the haiku test split (`H1-test-haiku.json`):

  | variant | R@1k tok | R@2k tok | R@4k tok |
  |---|---:|---:|---:|
  | hybrid | 0.615 | 0.732 | 0.830 |
  | hybrid_bm25 (H1) | 0.689 | 0.792 | 0.882 |
  | bm25 only | 0.752 | 0.808 | 0.864 |

  The full top-20 context is about 6.5k passage tokens. The open question
  is whether answers hold up when it is cut to ~2-4k by a score-gap rule.

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
- **Baseline (haiku stores, convs 5-9; `results/entity-resolution/`):**
  2,242 entities. 525 (23.4%) have a Haiku-confirmed duplicate, but session-
  header artefacts (H15) account for most of them; excluding those, 240
  entities (10.7%) have one ("low-impact exercises" / "low-impact
  exercise", "animal shelter" / "shelter"). Blocking precision against the
  labels: containment 0.26, token Jaccard 0.23, same-type mutual-nearest
  embedding 0.46 (39 flagged). Rules alone won't reach the predicted 0.95,
  so the adjudicator is required. Labels are Haiku on names + descriptions
  (some debatable, e.g. "photo of dogs" / "dogs running photo"), and recall
  is bounded by the blocking. `pairs-haiku.jsonl` (1,514 pairs) is the H9
  evaluation set.
- **Side finding (relation names):** the same relation under several names,
  e.g. `is_part_of` 565 / `part_of` 129, `participated_in` /
  `participates_in`, `conversed_at` / `conversation_at` / `talked_in`.
  Normalising relation names is a cheap separate step worth testing with H9.

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

### H15: Keep synthetic chunk headers out of entity extraction
- **Status:** proposed
- **Observation:** in the haiku stores, about 20% of Entity nodes (83-100 of
  420-527 per conversation) are session-header artefacts ("session 6
  conversation between audrey and andrew", "conversation session 25"),
  extracted from the dated header upstream's LoCoMo preprocessing puts on
  every window. They crowd the entity lane and are the bulk of the alias
  candidates in H9's data.
- **Claim:** passing the header to extraction as context ("this passage is
  from session 6, dated ...") rather than as extractable text removes those
  nodes, raises entity-lane precision, and leaves temporal recall unchanged
  (the date stays in the chunk text that is embedded and shown).
- **Metric:** session-like entity count; hybrid recall@5 on temporal
  (guard) and overall; entity-lane hit rate. Needs re-ingestion (new store).

## Tier T1/T2: need an LLM

These run through the Claude Code proxy (`enhanced/llm_proxy.py`).

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
