# Experimental protocol

Source: `ReGraph.md` §3. This file is the single reference for anything that affects
comparability. If a run deviates from it in any respect, that deviation must appear in the results
table.

## Benchmark

GraphQA, as introduced by G-Retriever (He et al., 2024). Each example is a triple $(G_i, q_i, A_i)$
with textual node and edge attributes. Graph reasoning is formulated as conditional language
generation: no task-specific prediction head, for any dataset.

| Dataset     | Examples | Avg. nodes | Avg. edges | Target                        | Metric   |
| ----------- | -------- | ---------- | ---------- | ----------------------------- | -------- |
| ExplaGraphs | 2,766    | 5.17       | 4.25       | `support` / `counter`     | Accuracy |
| SceneGraphs | 100,000  | 19.13      | 68.44      | short natural-language answer | Accuracy |
| WebQSP      | 4,737    | 1,370.89   | 4,252.37   | one or more entities          | Hit@1    |

- **ExplaGraphs** — small directed explanation graphs; nodes are commonsense concepts, edges are
  explanatory relations; decide whether two arguments support or counter each other.
- **SceneGraphs** — derived from GQA; nodes are objects and attributes, edges encode actions and
  spatial relations; questions range from direct attribute lookup to compositional relational
  reasoning.
- **WebQSP** — Freebase triples within two hops of the entities mentioned in each question;
  multiple valid entity answers, concatenated with `|` during teacher-forced training.

Reproduce these averages during preprocessing and compare; a mismatch means wrong split files.

## Splits

Use the **exact split indices released by G-Retriever**. Do not re-derive them.

- ExplaGraphs: 1,659 / 553 / 554 (train / val / test).
- SceneGraphs: official 60/20/20 over **image identifiers**, so questions about the same scene
  graph never cross subsets.
- WebQSP: the original RoG train/validation/test partition; the empty validation graph removed by
  the official preprocessing is also excluded here.

## Model configuration

| Component                 | Setting                                                                   |
| ------------------------- | ------------------------------------------------------------------------- |
| Backbone                  | meta-llama/Llama-3.1-8B-Instruct,**all original parameters frozen** |
| Attribute encoder         | `sentence-transformers/all-roberta-large-v1`                            |
| Graph encoder             | relation-aware 4-layer graph transformer, 1,024-d node memory             |
| Transition operator       | $P_i = \tilde D_i^{-1}(A_i + I)$                                        |
| Graph-query tokens        | 8                                                                         |
| Graph-reading rounds$T$ | 3                                                                         |
| Diffusion depth$K$      | 2                                                                         |
| Reader dimension          | 1,024                                                                     |
| Reader heads              | 8                                                                         |
| Dropout                   | 0.1                                                                       |
| LLM layer groups          | 4                                                                         |
| Reader / Fuse parameters  | shared across rounds                                                      |
| Trained modules           | graph encoder,$B_{\text{base}}$, reader, Fuse, graph→LLM projections   |

The graph is **not serialized into the LLM context**; the language input is the question, the
learnable graph-query tokens, and the answer boundary.

## Training

| Setting        | Value                                                                                |
| -------------- | ------------------------------------------------------------------------------------ |
| Loss           | answer-only next-token likelihood (question, query tokens, boundary, padding masked) |
| Optimizer      | AdamW                                                                                |
| Learning rate  | 1e-5                                                                                 |
| Weight decay   | 0.05                                                                                 |
| Schedule       | warm-up then cosine decay                                                            |
| Max epochs     | 10                                                                                   |
| Early stopping | patience 2                                                                           |
| Batch size     | 4                                                                                    |
| Checkpoint     | lowest validation loss                                                               |
| Scope          | a separate model per dataset                                                         |

No intermediate supervision exists for reading distributions, hop weights, fusion gates, or
reasoning states; every ReGraph component is learned end to end from the answer likelihood.

## Inference

Test-time input is $(G_i, q_i)$ only. All Read–Fuse–Replace rounds run during prefill
($F_0 \to \Gamma_0 \to F_1 \to \cdots \to \Gamma_{T-1} \to F_T$), followed by standard KV-cached
autoregressive decoding: **greedy, sampling disabled, stop at `EOS` or 32 generated tokens**,
matching G-Retriever.

## Reporting

Results table shape:

| Method                                                    | ExplaGraphs (Acc) | SceneGraphs (Acc) | WebQSP (Hit@1) |
| --------------------------------------------------------- | ----------------- | ----------------- | -------------- |
| Zero-shot Llama2-7b¹                                      | 56.50             | 39.74             | 41.06          |
| Prompt tuning (Llama2-7b, frozen)¹                        | 57.63 ± 2.43      | 63.41 ± 0.24      | 48.34 ± 0.64   |
| GraphToken (Llama2-7b, frozen)¹                           | 85.08 ± 5.51      | 49.03 ± 1.05      | 57.05 ± 0.74   |
| G-Retriever (Llama2-7b, frozen + PT)¹                     | 85.16 ± 0.92      | 81.31 ± 1.62      | 70.49 ± 1.21   |
| G-Retriever w/ LoRA (Llama2-7b)¹                          | 87.05 ± 3.29      | 86.83 ± 0.72      | 73.79 ± 0.70   |
| **ReGraph (Llama-3.1-8B-Instruct, frozen — this repo)**   | **92.42**²        | **51.83**⁴        | **62.22**³     |

¹ Copied from G-Retriever (He et al., 2024), arXiv:2402.07630 Table 3; their backbone is
Llama2-7b, while `ReGraph.md` §3 prescribes Llama-3.1-8B-Instruct for ReGraph — backbone
strength contributes to the gap, so read the comparison accordingly.
**All three rows are the corrected runs** (`runs/{dataset}/fix-seed0/`, 2026-08-20) with the Q18
double-dropout bug fixed. The pre-fix numbers were 92.42 ± 0.36 (3 seeds) / 52.25 / 62.47; the
fix is a correctness change with **no measurable effect on any dataset** (Δ = 0.00 / −0.42 /
−0.25 pp, all inside noise: SE is 1.11 pp on ExplaGraphs, 0.35 pp on SceneGraphs, 1.20 pp on
WebQSP). Both sets of predictions are kept in the run dirs.

² Single seed for the corrected run, 554/554 test examples, 100% non-empty; the pre-fix
three-seed spread was 92.06 / 92.78 / 92.42 = 92.42 ± 0.36. Best-val-loss checkpoint, greedy
decoding. `P` symmetrized for diffusion (Q2); WebQSP splits per Q16.
³ Single seed (protocol Q8), 1,628/1,628 test examples, 100% non-empty; auxiliary scores
F1 38.09 / precision 45.72 / recall 43.84. ReGraph **underperforms** G-Retriever's frozen
70.49 here despite the stronger backbone: it reads the full ~1,371-node graph through 8
query tokens with no retrieval and no serialized subgraph, and that compression is the
binding constraint on WebQSP (single-answer questions hit 51.7 vs 73.2 for multi-answer).
The graph path itself is heavily used — mean fusion gates 0.76/0.85/0.87 across rounds,
hop-0 mass ~0.65-0.72 with the rest through 1-2 hop diffusion — so the mechanism is alive;
the bottleneck is capacity, not a dead reader. Reported plainly per 09-evaluation.md §9.3.
⁴ Single seed (Q8), 20,025/20,025 test examples, 100% non-empty, all predictions in-vocabulary
single words (0 of 9,561 misses are case- or format-fixable). ReGraph **underperforms**
G-Retriever's frozen 81.31. Failure mode: fine-grained compositional/spatial errors —
yes/no questions score 56.2 (near chance), left/right relations frequently flipped — while
gates run at 0.47/0.97/0.96, i.e. the reader is maximally engaged but 8 query tokens cannot
carry per-object relational detail that G-Retriever hands the LLM as serialized subgraph text.
Together with WebQSP this delineates where the no-serialization design pays (small holistic
graphs: ExplaGraphs) and where it binds (many-node, detail-critical graphs).

Only the ReGraph row is produced by this repo. Comparison rows are **copied from the published
papers and cited** — do not re-run or reimplement baselines.

### Tuning log (2026-08-20) — SceneGraphs, accuracy on the first 1,500 test examples

Short runs used to choose a recipe. `N_B` = graph-query tokens, `W_O` = reader output-projection
init, gates = mean fusion gate per round at eval. Standard error on a difference here is ≈1.7
points, so sub-2-point gaps are noise.

| # | steps | N_B | lr | `W_O` | accuracy | gates | note |
|---|---|---|---|---|---|---|---|
| A | 3k | 8 | 1e-5 | zeros | 34.93 | 0.45 / 0.56 / 0.51 | protocol reference |
| B | 3k | 8 | 3e-4 | normal | 34.80 | 0.32 / **0.00** / **0.00** | graph path killed |
| C | 3k | 32 | 3e-4 | normal | 35.27 | 0.52 / **0.00** / **0.00** | graph path killed |
| D | 3k | 32 | 3e-4 | normal | 36.27 | 0.56 / **0.00** / **0.00** | `K=0`; graph path killed |
| H | 10k | 32 | 1e-5 ×`lr_mult` | normal | 36.33 | **0.01 / 0.04 / 0.03** | evidence path 20×, gate 1× — still killed |
| E | 10k | 32 | 1e-5 | zeros | 43.47 | 0.48 / 0.76 / 0.72 | gates healthy |
| G | 10k | 32 | 1e-5 | normal | 45.13 | 0.47 / 0.75 / 0.71 | gates healthy; best observed |
| I | 10k | 8 | 1e-5 | zeros | **43.67** | 0.49 / 0.80 / 0.77 | **the control** (re-run 2026-08-20) |
| E64 | 10k | 64 | 1e-5 | zeros | 43.00 | 0.42 / 0.76 / 0.67 | wider still, no gain |

**Verdict once the control ran: no configuration beats the protocol.** With 1,500 eval
examples, SE(difference) = 1.81 pp, so the 95% band is ±3.55 pp:

| arm | accuracy | vs control | significance |
|---|---|---|---|
| N_B=8 protocol (control) | 43.67 | — | — |
| N_B=32, zero-init | 43.47 | −0.20 pp (0.11 SE) | not significant |
| N_B=32, normal-init | 45.13 | +1.46 pp (0.81 SE) | not significant |
| N_B=64, zero-init | 43.00 | −0.67 pp (0.37 SE) | not significant |

Channel width does nothing across 8 → 32 → 64, and the earlier 43–45% readings were
step count, not the recipe. `configs/*_tuned.yaml` were therefore **deleted** rather than
shipped; `reader.w_o_init` and `train.lr_mult` remain available in code (defaults = protocol)
but neither is a demonstrated improvement.

Conclusions supported by this log:

1. **Raising the learning rate is harmful.** Every raised-lr arm drove the fusion gate to zero,
   i.e. the model learns to switch the graph off, with no accuracy benefit. Holding `fuse` at 1×
   does not help (row H): AdamW normalizes per-parameter step size, so a single scalar gate head
   moves decisively at any lr. This refutes the Q10/Q19 hypothesis that the protocol lr was the
   binding constraint.
2. **`w_o_init: normal` is safe only at the protocol lr** (G vs B/C/D/H).
3. **Channel width is not the lever** — settled by the control above. `N_B = 8`, 32 and 64 are
   indistinguishable at matched steps, so the readout bottleneck is not the *number* of evidence
   vectors. Combined with (1) and (2), every hyperparameter knob available within ReGraph's
   specified architecture has now been tried and none beats the protocol.
4. **What remains untested is the Q18 bug fix at full scale.** Every headline number
   (92.42 / 52.25 / 62.47) was produced with `R` dropped out twice. Full protocol reruns with
   the corrected single dropout are in `runs/{dataset}/fix-seed0/`; those are the corrected
   protocol numbers and the only outstanding source of improvement.
5. **Working hypothesis for the residual gap on SceneGraphs and WebQSP: it is architectural.**
   ReGraph must move an attribute like `white` through a convex average of node vectors injected
   into the mid-stack residual stream, whereas G-Retriever hands the LLM the same attribute as
   literal prompt tokens that its own attention can read at full bandwidth. No learning-rate,
   initialization, or width setting changes what that channel can express.

### Mechanism diagnostics — does the graph path actually carry graph information?

Measured on the trained checkpoints (2026-08-19). These qualify the accuracy table above and
should be read alongside it.

| Quantity | ExplaGraphs | SceneGraphs | WebQSP |
| --- | --- | --- | --- |
| H(S̃) as % of uniform (max entropy) | 90% | 21-25% | 73-76% |
| Cross-example cosine of injected `R` | 0.9994 | 0.706 | 0.903 |
| Example-**specific** share of `R` energy | **2.4%** | 52% | 31% |
| `W_O` dims carrying 90% of energy (of 1024) | 45 | 321 | 29 |
| ‖g·R‖ / ‖B_pre‖ (round 0) | 112% | 161% | 172% |
| Gold answer node's median rank under S̃ | — | — | **232 / 1371** |
| `W_Q`,`W_K` std vs. init | 1.00× | 0.99× | 1.00× |
| Reader/graph LayerNorm gains (init 1.0) | 1.0000 | 1.0002 | 1.0004 |
| Role-embedding norm vs. ‖h‖≈32 | 0.011 | 0.034 | 0.038 |

Readings:

1. **The ExplaGraphs result is not evidence that graph reading works.** The injected evidence
   `R` is ~98% identical across different graphs, i.e. functionally a learned bias — ReGraph
   there is soft-prompt tuning plus a constant. The 92.42 comes from the question text (which
   states both arguments verbatim) and the Llama-3.1-8B-Instruct backbone.
2. **On WebQSP the reader never localizes**: the gold answer node sits at median rank 232 of
   ~1,371 with only 1.5× uniform mass, so `R` is close to a mean-pooled graph summary — the very
   ablation baseline of §3.3, injected forcefully (gate 0.87) three times.
3. **The node-selection parameters barely moved from initialization** on any dataset. Two
   compounding causes: `W_O` is zero-initialized (04-reader.md §4.5), and since
   ∂L/∂S̃ = (∂L/∂R)(V_H W_O)ᵀ, the attention receives *exactly zero* gradient at step 0 and only
   `‖W_O‖`-scaled gradient thereafter (`W_O` reaches std 5e-4 to 2e-3 at lr 1e-5); the node
   softmax then divides per-node gradient by roughly n. Step budgets differ 18×
   (WebQSP 4,236 vs SceneGraphs 74,970), which matches which reader became selective.
4. **Query-role markers (§2.1) are effectively unused** — role-embedding norms are ~0.1% of ‖h‖.
5. On SceneGraphs the reader *did* learn to select (peaked S̃, 52% example-specific evidence,
   widest `W_O` channel) yet still scores 52.25, so its bottleneck is different: what a convex
   average of node vectors can express. See footnote ⁴.

These are diagnostics of the implemented spec, not proposed changes; no variant was built
(CLAUDE.md ground rule 7).

### ReGraph diagnostics (per dataset, filled as runs complete)

| Dataset | seed | best val loss (epoch) | mean gate r0/r1/r2 (last epoch) | notes |
| --- | --- | --- | --- | --- |
| ExplaGraphs | 0 | 0.1782 (1) | 0.48 / 0.53 / 0.51 | early stop at 3 |
| ExplaGraphs | 1 | 0.2073 (0) | 0.48 / 0.49 / 0.47 | early stop at 2 |
| ExplaGraphs | 2 | 0.2363 (0) | 0.45 / 0.45 / 0.44 | early stop at 2 |
| WebQSP | 0 | 1.4212 (3) | 0.90 / 0.94 / 0.95 | early stop at 5; test-time mean gates 0.76/0.85/0.87, hop dist ≈ [0.7, 0.15, 0.13] |
| SceneGraphs | 0 | 0.7012 (2) | 0.46 / 0.97 / 0.97 | early stop at 4; test-time mean gates 0.47/0.97/0.96; round-0 gate low, rounds 1-2 saturated — later reads dominate |
| ExplaGraphs | fix-seed0 | 0.1968 (1) | 0.55 / 0.62 / 0.56 | Q18-corrected; test 92.42 |
| SceneGraphs | fix-seed0 | 0.6999 (2) | 0.43 / 0.93 / 0.96 | Q18-corrected; test 51.83 |
| WebQSP | fix-seed0 | 1.4055 (3) | 0.92 / 0.95 / 0.95 | Q18-corrected; test Hit@1 62.22 |

- Seeds: the spec does not state a seed policy. Default to **3 seeds** on ExplaGraphs (small and
  cheap) reporting mean ± std, and single-seed runs on SceneGraphs and WebQSP, stating this
  explicitly in the table caption. Recorded in `docs/OPEN-QUESTIONS.md`.
- Always report alongside the metrics: mean fusion gate per round, mean hop distribution per round,
  and reading entropy. They are what distinguishes "the mechanism works" from "the numbers moved".
- Log wall-clock training time and peak GPU memory per dataset.

## Run log (this implementation, 2026-08-18, 1× RTX PRO 6000 96GB per run)

| Dataset | epochs run (best) | min/epoch | peak mem | eval |
| --- | --- | --- | --- | --- |
| ExplaGraphs ×3 seeds | 3-4 (0-1) | ~2.6 | 16.9 GB | 554 test in ~10 s |
| SceneGraphs | 5 (2) | ~42 | 16.9 GB | 20,025 test in ~4 min |
| WebQSP | 6 (3) | ~9 | 17.5 GB | 1,628 test in ~11 min |

KV-cache sanity (08-inference.md test 5): cached decoding is token-identical to full
recompute (24/24 ExplaGraphs examples; divergence appears only when generation is forced
past natural EOS, where bf16 near-ties between `<|eot_id|>` repetitions flip). Measured
per-token cost on an idle GPU (batch 4): cached 17.9 ms vs naive 26.3 ms on WebQSP-scale
graphs (1.5×), 17.5 vs 45.2 ms on ExplaGraphs (2.6×) — far from the doc's ~5× because
GraphQA prompts are only 25-90 tokens, so single-token decode is kernel-launch-bound and
the naive path's O(S) recompute is still tiny; the cached path does asymptotically less
work and wins increasingly with sequence length.

---

## Second benchmark: ogbn-arxiv (GraphTranslator, WWW 2024)

Added 2026-08-21. Chosen because GraphTranslator shares ReGraph's two hard constraints — the
LLM is **frozen** and the graph is **never serialized into the prompt** — so it is a
like-for-like comparison where G-Retriever (text channel) and G-reasoner (GPT-4o-mini, black
box) are not. Task: predict a paper's arXiv CS subject area, as open-ended text generation
with no classification head.

| Method | backbone | protocol | Top-1 |
| --- | --- | --- | --- |
| GraphTranslator¹ | ChatGLM2-6B, frozen | **zero-shot** | 28.48 (Top-3 37.62 / Top-5 39.87) |
| Majority class | — | — | 21.90 |
| **ReGraph (control, no alignment)** | Llama-3.1-8B-Instruct, frozen | **supervised** | **71.75** |

¹ Zhang et al., WWW 2024, Table 1.

**The protocol asymmetry is the headline caveat** (Q22): GraphTranslator is zero-shot, ReGraph
is trained on 20,000 ogbn-arxiv train nodes, and the backbones differ. These numbers are *not*
a like-for-like win and must not be presented as one. What they do establish is that ReGraph
behaves sensibly on this task — 71.75 sits in the range a well-tuned supervised GNN reaches on
ogbn-arxiv, against a 21.90 majority-class floor, with 100% legality and 37 of 40 categories
used.

Setup: 2-hop sampled ego-subgraphs (fanout 10, mean 45.5 nodes), node text = title + abstract
via all-roberta-large-v1, untyped citation edges, centre node tagged `source` (the first real
use of the §2.1 role vocabulary in this repo), evaluated on GraphTranslator's exact 4,000-node
subset. Best val loss 0.2783 at epoch 2, early stop at epoch 4, gates 0.24 / 0.66 / 0.69.

### Why arXiv works when SceneGraphs and WebQSP do not

This result sharpens rather than contradicts the earlier diagnosis. The graph→LLM channel is a
convex average of node vectors pushed through `W_O` into the mid-stack residual stream. That
channel evidently carries **coarse semantics** perfectly well — a 40-way subject decision is
~5.3 bits — but it cannot carry **identity or surface form**, which is what the failures needed:
naming a specific entity (WebQSP single-answer 51.7%), or binding one attribute to one object
(SceneGraphs colour 28.1%). One mechanism explains all four datasets:

| dataset | what the answer requires | result |
| --- | --- | --- |
| ExplaGraphs | nothing from the graph (question text suffices) | 92.42, but evidence 98% example-invariant |
| **arXiv** | **coarse topic of the centre node** | **71.75** |
| SceneGraphs | bind a specific attribute to a specific object | 51.83 |
| WebQSP | emit an exact entity surface form | 62.22 |

### Alignment pretraining (Q21): tested and NOT effective

| arm | Stage-1 | val loss | test Top-1 |
| --- | --- | --- | --- |
| control | none | 0.2783 | **71.75** |
| aligned | title alignment, gate pinned open | 0.2582 | **71.45** |

Δ = −0.30 pp on 4,000 examples (SE of a difference 1.01 pp) — **no effect**. Better validation
loss did not transfer to accuracy.

Stage-1 did do what it was designed to do: `W_O` std reached 0.01318 (7× the ceiling of the
GraphQA plateau) and `W_Q` moved off its initialization for the first time in this project
(0.02347 vs 0.01976 init). The projector is trainable — it just does not change the outcome.

**Why, and a correction to earlier readings of the fusion gate.** Measuring the *effective*
injection rather than the gate alone:

| arm | gate (r0/r1/r2) | ‖R‖ | ‖g·R‖ / ‖B_pre‖ |
| --- | --- | --- | --- |
| control | 0.24 / 0.67 / 0.69 | 11.3 | 52% / 89% / 40% |
| aligned | 0.06 / 0.08 / 0.18 | **654** | **659% / 147% / 143%** |

The aligned arm's evidence is 58× larger in norm and the gate closes to compensate. The two
arms deliver *comparable or larger* graph signal into the residual stream and reach the same
accuracy. Alignment changed the internal parameterization, not the information content.

**This invalidates a claim made earlier in this file.** The 2026-08-20 tuning log states that
raised-lr arms "drove the fusion gate to zero, i.e. the model switches the graph off". The
accuracy findings there stand (no configuration beat the protocol), but the *mechanism* claim
does not: a low gate paired with a large ‖R‖ is normalization, not shutdown, and ‖R‖ was never
measured in that sweep. Gate means must always be read together with ‖R‖; on their own they are
not a health signal, contrary to `05-fuse.md` §5.1.

**Standing conclusion.** ReGraph's accuracy on a task is set by what its convex-average channel
can carry — coarse semantics yes, identity and surface form no — and not by how well the
graph→LLM projection is trained. Two independent interventions (every hyperparameter, then
alignment pretraining) failed to move it, while changing the *task* moved it from 51.83 to 71.75.
