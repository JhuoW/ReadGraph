# Open questions

Living log of everything `ReGraph.md` does not fully determine. Each entry: the question, the
default taken, and where it is configured. **Append to this file whenever you have to decide
something the spec does not state** — and say so in your response to the user.

Status: `OPEN` = needs a decision from the author; `DEFAULTED` = a documented default is in use.

---

### Q1 — Llama3.1-8B base or Instruct? · `DEFAULTED`

meta-llama/Llama-3.1-8B-Instruct

*2026-08-18:* `configs/default.yaml` shipped with `llm.name: meta-llama/Llama-3.1-8B` (base),
contradicting `ReGraph.md` §3 ("meta-llama/Llama-3.1-8B-Instruct backbone") and this entry.
Per CLAUDE.md ground rule 1 the spec wins — config corrected to the Instruct checkpoint.
The prompt stays plain text (no chat template), matching G-Retriever's usage.

### Q2 — Directedness of $P$ for diffusion · `DEFAULTED`

$P = \tilde D^{-1}(A+I)$; the spec does not say whether $A$ is symmetrized. All three datasets have
directed edges.
**Default:** symmetrize for diffusion only; the graph encoder keeps the original directed edges.
Config: `graph.symmetrize_for_diffusion: true`. Report the setting in results.

### Q3 — Answer-start token $y_0$ · `DEFAULTED`

The spec uses a single token $y_0$; a readable boundary string is usually several tokens.
**Default:** boundary string `"\nAnswer:"` treated as an $n_0$-token boundary block, shifting the
answer offset to $j_s = N_q + N_B + n_0 + (s-1)$. Set `data.boundary_tokens = 1` to match the spec
exactly. Config: `data.answer_boundary`.

### Q4 — Per-head vs shared hop weights $\alpha^t$ · `DEFAULTED`

The spec defines $\alpha^t \in \mathbb R^{N_B \times (K+1)}$ for the single-head reader and says
multi-head uses independent Q/K/V per head, without specifying $\alpha$.
**Default:** per-head hop weights. Config: `reader.shared_hop_weights: false`.

### Q5 — Warm-up length · `DEFAULTED`

"Warm-up followed by cosine decay" with no length given.
**Default:** 5% of total steps. Config: `train.warmup_ratio: 0.05`.

### Q6 — Graph encoder details · `DEFAULTED`

"Relation-aware four-layer Graph Transformer", 1,024-d output; head count and normalization
placement unspecified.
**Default:** PyG `TransformerConv` with `edge_dim=1024`, 4 heads, residual + post-LayerNorm + GELU,
dropout 0.1. Config: `graph_encoder.*`.

### Q7 — Role assignment procedure · `DEFAULTED`

The spec defines the role set but not how references are resolved. It requires assignment only when
resolution is unambiguous.
**Default:** normalized exact string match for `mentioned`; `source` from WebQSP's `q_entity`;
`target` unused; ambiguous matches → all `none`. Config: `data.roles.*`. Log the non-`none` rate per
dataset.

### Q8 — Seed policy · `DEFAULTED`

Not specified.
**Default:** 3 seeds on ExplaGraphs (mean ± std), single seed on SceneGraphs and WebQSP; stated in
the table caption. See `docs/experimental-protocol.md`.

### Q9 — SceneGraphs evaluation cost · `OPEN`

100,000 examples with a 20% test split means ~20,000 generations per evaluation. Confirm with the
author whether full-test evaluation is intended at every checkpoint or only at the end, and whether
a fixed evaluation subset is acceptable for validation-loss-based model selection.
(Validation *loss* is cheap — this concerns generation-based metrics only.)

### Q10 — Learning rate for from-scratch modules · `OPEN`

`lr = 1e-5` is specified, but every trainable module here is randomly initialized rather than
fine-tuned. If training underfits, flag it with evidence (loss curves, gate means) rather than
changing the value — any change breaks comparability with the reported protocol.
*2026-08-18:* the 16-example ExplaGraphs overfit test reaches loss ≈ 0.0000 within 25 steps at
lr 1e-5, so the specified rate is not a blocker there.

### Q11 — Exact prompt strings per dataset · `DEFAULTED`

The spec fixes only "question + graph-query tokens + answer boundary". Question *content* follows
G-Retriever exactly (ExplaGraphs: `Argument 1: ...\nArgument 2: ...\nQuestion: Do argument 1 and
argument 2 support or counter each other? ...`; SceneGraphs/WebQSP: the raw question).
**Default:** `data.prompt_template = "Question: {question}\n"` with boundary `"\nAnswer:"`;
ExplaGraphs overrides the template to `"{question}\n"` because its question already carries its own
`Question:` line. Rendered text matches G-Retriever byte-for-byte for ExplaGraphs and SceneGraphs;
WebQSP differs by one newline (ours `Question: q\n\nAnswer:` vs theirs `Question: q\nAnswer: `).
The answer is tokenized as `" {answer}"` + EOS. Config: `data.prompt_template`,
`data.answer_boundary`.

### Q12 — Answer-token truncation during teacher forcing · `DEFAULTED`

The spec does not bound answer length at training time; G-Retriever truncates labels to
`max_new_tokens = 32` before appending EOS. **Default:** the same — `data.max_answer_tokens: 32`
(+EOS). Only WebQSP multi-entity answers ever hit the cap. Questions get a safety cap
`data.max_question_tokens: 512` (never reached in practice).

### Q13 — Attribute-embedding storage · `DEFAULTED`

The spec does not constrain how pre-computed embeddings are stored. Naive per-graph storage of
WebQSP full graphs is ~100 GB fp32. **Default:** deduplicate node/edge texts per dataset, encode
each unique text once, store one fp16 memmap (`emb.f16.npy`) plus per-graph int32 id arrays; fp32
is restored at load. sbert outputs are L2-normalized so fp16 quantization error is ~1e-3 and
identical for all occurrences of a text. Cache dir embeds the encoder name (invalidates on swap).

### Q14 — Where `inputs_embeds` are assembled · `DEFAULTED`

`01-data.md` lists `inputs_embeds` among collate outputs. Embedding lookup needs the frozen
`embed_tokens`, which lives on GPU. **Default:** collate returns `input_ids` (+ `b_positions` with
the placeholder id in the B slots); `ReGraph.build_inputs_embeds` embeds and scatters `b_base` on
device, exactly as `01-data.md`'s "Prompt assembly" paragraph describes. Semantics unchanged.

### Q15 — Fully padded query rows in the causal mask · `DEFAULTED`

Right-padded rows have query positions with no attendable key (softmax over -inf → NaN, which
poisons gradients even at zero-weight positions). **Default:** padded query rows attend to their
own slot; real rows are untouched (this mirrors HF's `unmask_unattended` fix). Implementation:
`ReGraph._additive_causal_mask`, used identically in training, prefill, and decode so all three
paths share one numeric behavior.

### Q17 — `W_O` zero-init starves the reader's attention · `DEFAULTED` (protocol) / deviation available

`04-reader.md` §4.5 mandates zero-initialized `W_O` so that `R^t = 0` at step 0 and the model
starts exactly at frozen-LLM behavior. `ReGraph.md` itself says nothing about initialization, so
per CLAUDE.md ground rule 1 this is a doc-level default, not spec.

It has an unintended consequence. Since `R = S̃ V_H W_O`, we have
`∂L/∂S̃ = (∂L/∂R)(V_H W_O)ᵀ`, so with `W_O = 0` the node-selection weights `W_Q`/`W_K` receive
**exactly zero** gradient at step 0 and only `‖W_O‖`-scaled gradient afterwards. Measured on all
three trained checkpoints: `W_Q`/`W_K` never leave their initialization (1.00×, 0.99×, 1.00× of
init std) and WebQSP's reading stays at 73% of uniform entropy with the gold node at median
rank 232/1371.
**Default:** unchanged (`reader.w_o_init: zeros`) for protocol runs.
**Deviation:** `reader.w_o_init: normal` with `reader.w_o_init_std: 1.0e-3` opens the attention
gradient path from step 0, trading the exact identity-at-init property for a trainable reader.
Used only by `configs/*_tuned.yaml`. Regression tests in `tests/test_reader.py` pin both
behaviours.

*2026-08-20 empirical note.* `normal` is only safe at the protocol learning rate. Paired with a
raised lr it makes `R` a fast-moving random signal, the fusion gate correctly learns to suppress
it, and since the gradient reaching `W_O` is scaled by that gate, closing it re-severs the path
this option was meant to open. Measured on SceneGraphs at 10k steps: `normal` + lr 1e-5 keeps
gates at [0.47, 0.75, 0.71], while `normal` + raised lr collapses them to [0.01, 0.04, 0.03].

### Q18 — Evidence dropout was applied twice · `FIXED` (spec-fidelity bug)

`ReGraph.md` §2.4 states exactly one dropout on the evidence, inside Fuse:
`B_post = B_pre + Diag(g) Dropout(R)`. The implementation additionally applied
`reader.dropout` to `R` in `model.py` before calling Fuse, so the graph signal was dropped at an
effective rate of `1 - 0.9² = 0.19` rather than 0.10. **Fixed** — Fuse owns the single
`Dropout(R)`; the `reader.dropout` config key now has no output-side effect and is documented as
such. This is a correction toward the spec, not a deviation, and applies to protocol runs too, so
protocol numbers reported before 2026-08-20 were produced with the double dropout.

*Re-measured 2026-08-20 on all three datasets (`runs/{dataset}/fix-seed0/`):* correctness only —
92.42 vs 92.42, 51.83 vs 52.25, 62.22 vs 62.47. Every delta is inside noise, so the corrected
runs are now the reported numbers but no conclusion changes.

### Q19 — Learning rate for from-scratch modules, revisited · `DEVIATION` (author-approved)

Follow-up to Q10. `ReGraph.md` §3.3 specifies `lr = 1e-5`; G-Retriever uses the same value, but
their graph module only has to emit one summary token because the retrieved subgraph is
serialized into the prompt and carries the information. In ReGraph the reader is the *only*
channel, so it must learn to attend, and the evidence in Q17 shows it does not at `1e-5` within
the protocol's step budget (WebQSP gets just 4,236 updates).
**Default:** unchanged (`train.lr: 1.0e-5`, `train.lr_mult: {}`) — `configs/{dataset}.yaml`
reproduce the protocol exactly.
**Deviation:** `configs/{dataset}_tuned.yaml` raise the learning rate (and may extend the epoch
budget). `train.lr_mult` allows per-module multipliers keyed by parameter-name prefix.
Any run using these configs is **not** protocol-comparable and must be reported on its own row.

*2026-08-20 empirical note — raising the learning rate is HARMFUL and the tuned configs no longer
do it.* Every raised-lr variant killed the graph path (gates → 0) with no accuracy gain; see the
tuning log in `docs/experimental-protocol.md`. Holding `fuse` at 1× via `lr_mult` does **not**
rescue it: AdamW normalizes per-parameter step size, so a one-scalar gate head moves decisively
regardless of its lr. The surviving lever is channel width (Q20), at the protocol lr.

### Q20 — Number of graph-query tokens · `DEVIATION` (author-approved, unvalidated)

`ReGraph.md` §3.3 fixes `N_B = 8`. Since the readout — not node selection — is the measured
bottleneck, widening the graph→LLM channel is the most direct lever: with more evidence vectors
the LLM's own attention does the selecting instead of the reader's barely-trained attention.
**Default:** unchanged (8). **Deviation:** `model.num_query_tokens: 32` in `configs/*_tuned.yaml`.
**Status: not yet validated** — the matched 10k-step protocol control never ran (the backbone
cache was deleted mid-sweep), so the apparent gain is not attributable to `N_B` yet.

### Q16 — WebQSP example count: 4,737 vs the released splits · `DEFAULTED`

`ReGraph.md` §3.1 (and the G-Retriever paper's table) lists WebQSP as 4,737 examples with
avg 1,370.89 nodes / 4,252.37 edges. The **exact split indices released by G-Retriever**
(which §3.1 mandates) contain train 2,826 + val 246 + test 1,628 = 4,700, minus the empty
validation graph at concatenated index 2937 → **4,699 usable examples (2,826 / 245 / 1,628)**.
Evidence we hold identical data: our per-graph averages over the 4,699 kept examples
(1,371.18 / 4,253.27) reproduce the published averages *exactly* when the dropped empty graph
is included in the denominator (1,371.18 × 4699/4700 = 1,370.89; 4,253.27 × 4699/4700 = 4,252.37).
4,737 is the original WebQSP question count quoted in dataset tables.
**Default:** follow the released split files (the spec's own mandate); expected stats in
`configs/webqsp.yaml` updated accordingly. Preprocessing initially *stopped and reported* on this
mismatch, per docs/components/01-data.md.

### Q21 — Graph→LLM alignment pretraining · `DEVIATION` (author-approved, under test)

`ReGraph.md` §3.2 states that "GraphQA provides no intermediate supervision ... all ReGraph
components are learned end to end from the final answer likelihood", and `07-training.md`
forbids auxiliary losses. Measurements say that objective is insufficient to train the
graph→LLM projection:

- across all three GraphQA checkpoints `W_O`'s std stays at 5e-4..2e-3 and `W_Q`/`W_K` never
  leave their initialization scale (1.00×);
- the reading distribution *degrades* during training — WebQSP gold-node median rank goes
  from 37 at init (9.5× better than random) to 301 trained (1.4×);
- every hyperparameter (lr, `lr_mult`, `w_o_init`, `K`, `N_B` ∈ {8,32,64}) is a measured null.

Pointing at the right node earns no loss reduction if the frozen decoder cannot read an
identity out of a convex average, so the optimizer spends the channel elsewhere. GraphTranslator
(WWW 2024), BLIP-2 and LLaVA all address the same problem with a projector-alignment stage
before task training.

**Deviation:** a Stage-1 pass (`configs/arxiv_align.yaml`) that keeps the LLM frozen and trains
the same graph-side modules to make the evidence tokens reproduce the centre node's *title* —
a dense, per-example, surface-form target. Stage 2 (`configs/arxiv.yaml --init-from`) then
fine-tunes on the task. Applied only to the arXiv benchmark; the GraphQA protocol runs are
untouched.

**Control:** the same Stage-2 run without `--init-from`. Report both rows so any gain is
attributable to the alignment rather than to arXiv being an easier task.

### Q22 — arXiv evaluation protocol vs GraphTranslator · `DEFAULTED`

GraphTranslator's ArXiv numbers (Top-1 28.48 / Top-3 37.62 / Top-5 39.87) are **zero-shot**:
its Translator is aligned on graph-text pairs and never trained on ogbn-arxiv labels. ReGraph
as specified is trained per dataset with answer supervision.
**Default:** report ReGraph supervised, on GraphTranslator's exact 4,000-node test subset, and
state the protocol asymmetry next to every number. Also documented: each example is a sampled
2-hop ego-subgraph (≤111 nodes, mean 45.5), not the full 169k-node graph, and training is
subsampled to 20,000 of the 90,941 train nodes (`dataset.max_train`).

### Q23 — Token channel: serializing the graph *in addition to* reading it · `DEVIATION` (author-approved) · **the one intervention that worked**

`ReGraph.md` §3.2 keeps the graph out of the LLM context, and every prior attempt to improve
SceneGraphs within that constraint failed (eight interventions, none significantly positive —
see the tuning log). The measured reason: the convex-average channel can relay semantics already
present in node features but cannot bind attributes to objects or compute relations.

**Deviation:** `data.serialize_graph: true` additionally renders the graph as CSV text into the
prompt (G-Retriever's `desc` format), while the Read-Fuse-Replace rounds stay active. Both
channels run. Config: `configs/scene_graphs_dual.yaml`.

**Result (3,000 steps, 1,500 eval examples):**

| arm | accuracy | train loss |
| --- | --- | --- |
| token channel + ReGraph (T=3) | **81.93** | 0.3229 |
| token channel only (T=0) | 74.93 | 0.5129 |
| ReGraph alone (protocol) | 34.93 | 0.9182 |

**A − B = +7.00 pp = 4.6 SE** on SceneGraphs. Extended to four datasets (2026-08-23), the
effect is large everywhere but **not consistent in sign**:

| dataset | reader alone | A: text+reader | B: text only | A − B |
| --- | --- | --- | --- | --- |
| SceneGraphs | 51.83 | 81.93 | 74.93 | +7.00 (4.7σ) |
| ExplaGraphs | 92.42 | 86.82 | 81.05 | +5.77 (2.6σ) |
| NLGraph connectivity | 49.33 | 87.87 | 75.74 | +12.13 (4.3σ) |
| NLGraph cycle | 52.88 | 75.92 | 90.58 | **−14.66 (−3.9σ)** |

Two generalizations were made from partial data and both were falsified by the next dataset:
"iterative reading always adds on top of serialization" (killed by cycle) and "the reader helps
iff it has standalone signal" (killed by connectivity, whose reader is also at chance yet gains
+12.13). **Standing conclusion: the token channel changes results substantially on every
dataset, but the sign is dataset-dependent and unpredicted by current evidence.** An untested
observation worth a purpose-built experiment: connectivity has explicit source/target anchors
for diffusion to propagate from, cycle is anchor-free.

Two implementation notes that matter for anyone reproducing this:
- The full graph is serialized, not a PCST subgraph: measured on SceneGraphs, PCST
  (topk=3, topk_e=3) retains the gold answer string in only 37.3% of examples vs 81.7% for the
  full graph, so retrieval caps the ceiling below the existing baseline.
- The graph text must be budgeted, **not** the question. Prepending the graph and truncating the
  whole prompt silently drops the question on long graphs (SceneGraphs p90 ≈ 1,700 tokens),
  leaving the model to answer blind on ~10-15% of examples. `GraphQADataset.__getitem__`
  truncates the graph and always appends the question intact.
