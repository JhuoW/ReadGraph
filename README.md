# ReGraph

Implementation of **ReGraph: Iterative Graph Reading for Open-Ended Reasoning with LLMs**
(spec of record: [ReGraph.md](ReGraph.md); per-component contracts in
[docs/components/](docs/components/)).

ReGraph interleaves `Read → Fuse → Replace` graph-reading rounds between groups of frozen
Llama-3.1-8B-Instruct layers. Eight learnable graph-query tokens sit in the LLM sequence after the
question; at each of the 3 rounds they query a node memory produced by a relation-aware graph
transformer through a **Topology-Diffused Graph Reader** (query-conditioned polynomial graph filter
over semantic relevance, K=2 hops), and the retrieved evidence is written back through a scalar
**gated residual fusion**. The graph is never serialized into the prompt. At inference all rounds
run once during prefill; decoding is standard KV-cached greedy decoding.

## Setup

```bash
pip install -r requirements.txt   # transformers is pinned: the manual layer loop
pip install -e .                  # is written against 4.57.3's modeling_llama.py
```

Raw data (see `docs/components/01-data.md`): copy G-Retriever's
`train_dev.tsv` → `data/raw/expla_graphs/`, `questions.csv` + `sceneGraphs.zip` →
`data/raw/scene_graphs/`. WebQSP loads from HF (`rmanluo/RoG-webqsp`).

## Pipeline

```bash
# 1. preprocess (builds graphs, dedup sbert embeddings, splits, roles; verifies protocol stats)
python -m regraph.data.preprocess --config configs/expla_graphs.yaml
python -m regraph.data.preprocess --config configs/scene_graphs.yaml
python -m regraph.data.preprocess --config configs/webqsp.yaml

# 2. train (one model per dataset; run dir = runs/<dataset>/<run_name>)
python -m regraph.train --config configs/expla_graphs.yaml run_name=seed0 [seed=0]

#    configs/<dataset>.yaml       = ReGraph.md §3.3 protocol, exactly (comparable numbers)
#    (a tuned-config variant was tried and removed: no setting beat the protocol — see
#     the 2026-08-20 tuning log in docs/experimental-protocol.md)

# 3. evaluate best checkpoint (greedy, ≤32 new tokens; writes predictions_test.jsonl,
#    metrics_test.json, readings_test.json)
python -m regraph.eval --config configs/expla_graphs.yaml \
    --ckpt runs/expla_graphs/seed0/best.pt --split test
```

## Experimental results

All ReGraph numbers below are produced by this repo. **Every baseline number is copied from the
cited paper and never re-run** (`docs/components/09-evaluation.md` §9.3). Full protocol,
diagnostics and caveats: [docs/experimental-protocol.md](docs/experimental-protocol.md);
every deviation from the spec: [docs/OPEN-QUESTIONS.md](docs/OPEN-QUESTIONS.md).
Which of these results constitute state of the art, and under which admission criteria:
[EVAL.md](EVAL.md).

ReGraph configuration throughout: Llama-3.1-8B-Instruct with **all LLM parameters frozen**,
8 graph-query tokens, 3 reading rounds, diffusion depth K=2, ~32.7M trainable parameters, and
**the graph is never serialized into the prompt**.

### Performance comparison

All baseline numbers are **copied from the cited papers and never re-run**
(`docs/components/09-evaluation.md` §9.3). Rows marked **(ours)** are produced by this repo.
Because the cited works use different backbones and different graph inputs, both are given as
columns — the comparison is not valid without them. `†` marks a row that is not
protocol-comparable to the block it sits in; see the notes under each table.

#### Table 1 — GraphQA (ExplaGraphs / SceneGraphs / WebQSP)

| #  | Method                                      | Backbone          | LLM    | Graph input           | ExplaGraphs     | SceneGraphs     | WebQSP         |
| -- | ------------------------------------------- | ----------------- | ------ | --------------------- | --------------- | --------------- | -------------- |
|    | *Inference only*                          |                   |        |                       |                 |                 |                |
| 1  | Zero-shot ᵃ                                | Llama2-7b         | frozen | none                  | 56.50           | 39.74           | 41.06          |
| 2  | Zero-shot (base) ᵇ                         | Llama-3.2-3B      | frozen | none                  | 13.5            | 33.1            | 32.7           |
| 3  | Zero-shot (chat) ᵇ                         | Llama-3.2-3B      | frozen | none                  | 52.6            | 50.7            | 53.4           |
| 4  | KAPING ᵇ                                   | Llama-3.2-3B      | frozen | text                  | 62.2            | 43.7            | 52.6           |
|    | *Tuning without graph structure*          |                   |        |                       |                 |                 |                |
| 5  | Prompt tuning ᵃ                            | Llama2-7b         | frozen | none                  | 57.63 ±2.43    | 63.41 ±0.24    | 48.34 ±0.64   |
| 6  | Prompt tuning ᵇ                            | Llama-3.2-3B      | frozen | none                  | 60.2            | 58.3            | 57.9           |
| 7  | LoRA ᵇ                                     | Llama-3.2-3B      | tuned  | text                  | 88.9            | 85.3            | 71.1           |
|    | *Tuning with graph structure*             |                   |        |                       |                 |                 |                |
| 8  | GraphToken ᵃ                               | Llama2-7b         | frozen | vector                | 85.08 ±5.51    | 49.03 ±1.05    | 57.05 ±0.74   |
| 9  | KG-Adapter ᵇ                               | Llama-3.2-3B      | frozen | vector                | —              | —              | 68.7           |
| 10 | GRAG ᵇ                                     | Llama-3.2-3B      | frozen | text+vector           | 88.9            | —              | 68.9           |
| 11 | G-Retriever ᵃ                              | Llama2-7b         | frozen | text+vector           | 85.16 ±0.92    | 81.31 ±1.62    | 70.49 ±1.21   |
| 12 | G-Retriever w/ LoRA ᵃ                      | Llama2-7b         | tuned  | text+vector           | 87.05 ±3.29    | 86.83 ±0.72    | 73.79 ±0.70   |
| 13 | GRAFF ᵇ                                    | Llama-3.2-3B      | frozen | text+vector           | **92.5**  | **90.2**  | **72.2** |
|    | *This work*                               |                   |        |                       |                 |                 |                |
| 14 | **ReGraph (ours)**                    | Llama-3.1-8B-Inst | frozen | **vector only** | **92.42** | 51.83           | 62.22          |
| 15 | **ReGraph + token channel (ours)** † | Llama-3.1-8B-Inst | frozen | text+vector           | 86.82           | **89.41** | n/a ‡         |
| 16 | *token channel only, no reader (ours)*    | Llama-3.1-8B-Inst | frozen | text                  | 81.05           | 74.93 †        | n/a ‡         |

ᵃ He et al., NeurIPS 2024 (arXiv:2402.07630) Table 3. ᵇ Chaudhary et al., Findings of EACL 2026,
Table 1 — evaluated on **PCST-retrieved subgraphs** (their Table 2: WebQSP 8.39 avg nodes,
SceneGraph 8.21), whereas rows 14-15 read the **full** graph (WebQSP ~1,371 nodes).

† Only the **SceneGraphs cell of row 16** is a short run (3,000 steps, scored on 1,500 examples);
every other cell in rows 15-16 is complete. Row 15's SceneGraphs figure is the finished 6-epoch
run over all 20,025 test examples (best validation loss at epoch 3; `runs/scene_graphs/dual-full`).

**Do not subtract row 16 from row 15 on SceneGraphs** — the two differ in training length *and*
evaluation set, so the difference is not a reader-contribution measurement. The matched A/B is
81.93 vs 74.93, both at 3,000 steps on the same 1,500 examples (**+7.00 pp, 4.6 SE**), reported in
full below; a full-scale no-reader control has not been run.

Rows 15-16 **violate `ReGraph.md` §3.2** by serializing the graph into the prompt — the
"augmentation" configuration, not ReGraph as specified.

‡ WebQSP cannot take a token channel: serializing its full graph costs **62,703 tokens on
average** (max 110,458), against Llama-3.1's 128k limit and far past where memory and speed hold
up. G-Retriever and GRAFF avoid this with PCST retrieval (8.39 nodes), but our port of their
released retrieval — verified to match their code example-for-example — keeps the gold answer in
only 13-15% of WebQSP examples versus 95% for the full graph, so it would cap the ceiling below
the existing 62.22.

**Row 15 minus row 16 is the load-bearing comparison**: +5.77 pp on ExplaGraphs (2.62 SE) and
+7.00 pp on SceneGraphs (4.6 SE). On two independent datasets, iterative graph reading adds
significantly *on top of* serialization — it is not subsumed by it. Note also that the token
channel is **not** universally good: on ExplaGraphs it costs 11.4 pp on its own (row 16 vs row
14), because that task is solvable from the question text and the serialized graph is a
distraction. It helps exactly where the graph is load-bearing.

Row 14 is the headline protocol result: ReGraph beats every baseline on ExplaGraphs while using
**no text channel at all**, and loses clearly on SceneGraphs and WebQSP. The mechanism behind
that split is analysed below.

#### Table 2 — ogbn-arxiv (GraphTranslator benchmark)

| Method                                 | Backbone                  | Protocol             | Top-1           | Top-3           | Top-5           |
| -------------------------------------- | ------------------------- | -------------------- | --------------- | --------------- | --------------- |
| Majority class                         | —                        | —                   | 21.90           | —              | —              |
| GraphTranslator ᶜ                     | ChatGLM2-6B, frozen       | zero-shot            | 28.48           | 37.62           | 39.87           |
| **ReGraph (ours)**               | Llama-3.1-8B-Inst, frozen | **supervised** | 71.75           | 92.58           | 96.40           |
| **ReGraph (ours), 3B**           | Llama-3.2-3B-Inst, frozen | **supervised** | **72.28** | **93.23** | **96.90** |
| ReGraph + alignment pretraining (ours) | Llama-3.1-8B-Inst, frozen | supervised           | 71.45           | 92.28           | 96.65           |

The 3B backbone matches or slightly exceeds the 8B one on all three ranking metrics (+0.50 / +0.65
/ +0.50), none of them significant (0.7-1.8 SE, n=4,000) — the backbone-insensitivity signature
discussed under "Backbone sensitivity" below. Top-1 is quoted from generation; the likelihood-rank
Top-1 is 71.60 (8B) and 72.10 (3B).

ᶜ Zhang et al., WWW 2024, Table 1. **Protocols differ** — GraphTranslator is zero-shot, ReGraph
is trained on 20,000 ogbn-arxiv nodes with a stronger backbone, so this is not a like-for-like
win. Evaluated on GraphTranslator's exact 4,000-node test subset; each example is a sampled
2-hop ego-subgraph (mean 45.5 nodes).

#### Table 3 — NLGraph (structural reasoning)

Accuracy by difficulty (Easy / Medium / Hard / Avg).

| Method                                      | Backbone          | Connectivity                           | Cycle                                  |
| ------------------------------------------- | ----------------- | -------------------------------------- | -------------------------------------- |
| Random                                      | —                | 50.00 / 50.00 / 50.00 / 50.00          | 50.00 / 50.00 / 50.00 / 50.00          |
| Zero-shot ᵈ                                | text-davinci-003  | 83.81 / 72.75 / 63.38 / 71.31          | 50.00 / 50.00 / 50.00 / 50.00          |
| Few-shot ᵈ                                 | text-davinci-003  | 93.75 / 83.83 / 76.61 / 84.73          | 80.00 / 70.00 / 61.00 /**70.33** |
| CoT ᵈ                                      | text-davinci-003  | 94.32 / 82.17 / 77.21 / 84.57          | 84.67 / 63.33 / 53.25 / 66.75          |
| CoT+SC ᵈ                                   | text-davinci-003  | 93.18 / 84.50 / 82.79 /**86.82** | 82.00 / 63.67 / 53.50 / 66.39          |
| **ReGraph (ours)**                    | Llama-3.1-8B-Inst | 66.07 / 47.96 / 43.70 /**52.58** | 64.00 / 49.53 / 54.24 /**55.92** |
| **+ token channel (ours)** ᶠ         | Llama-3.1-8B-Inst | 91.07 / 86.22 / 89.08 /**88.79** | 76.00 / 72.90 / 81.36 /**76.75** |
| *token channel only, no reader (ours)* ᶠ | Llama-3.1-8B-Inst | 87.50 / 72.45 / 75.63 /**78.53** | 88.00 / 91.59 / 89.83 /**89.81** |

ᶠ Edge list serialized into the prompt (the setting NLGraph's own baselines use), **plus**
supervised training — not protocol-comparable to the prompting rows above. Note the best cycle
configuration has the graph reader **switched off**.

ᵈ Wang et al., NeurIPS 2023, Table 2 (standard set). Their baselines are **prompting with the
edge list in the prompt**; ReGraph is **supervised** and reads topology through the encoder —
an asymmetry that favours ReGraph. It is nonetheless **at chance** on both tasks.

The four numbers per task are **Easy / Medium / Hard / Avg**, NLGraph's difficulty subsets, split
by graph size: Easy = 5-10 nodes, Medium = 11-25, Hard = 26-35 (their Table 1; connectivity
352/1200/680, cycle 150/600/400). **`Avg` is the unweighted mean of the three subsets**, not the
overall accuracy — verified by reproducing 4 of the 5 baseline rows exactly from their subset
numbers (their ZERO-SHOT row computes to 73.31 rather than the printed 71.31, apparently a typo).
ReGraph's Avg here follows the same convention; its size-weighted overall accuracy is 49.33
(connectivity) and 52.88 (cycle), which is what `scripts/nlgraph_report.py` also prints.

#### Table 4 — Cora (LLaGA / GraphGPT lineage, text-attributed)

Node classification as open-ended generation of the topic name (no classification head).
TAPE's 60/20/20 split, 2-hop ego-subgraphs (mean 17 nodes), 542 test examples.

| Method                       | Backbone                  | Cora (Acc)      |
| ---------------------------- | ------------------------- | --------------- |
| GCN ᵉ                       | —                        | 88.93           |
| GraphSAGE ᵉ                 | —                        | 88.89           |
| GAT ᵉ                       | —                        | 88.97           |
| SGC ᵉ                       | —                        | 87.97           |
| SAGN ᵉ                      | —                        | **89.19** |
| NodeFormer ᵉ                | —                        | 88.23           |
| GPT-3.5 (general setting) ᵉ | GPT-3.5                   | 71.75           |
| LLaGA-ND-7B ᵉ               | Vicuna-7B, frozen         | 88.86           |
| LLaGA-HO-7B ᵉ               | Vicuna-7B, frozen         | **89.22** |
| **ReGraph (ours)**     | Llama-3.1-8B-Inst, frozen | **86.72** |

ᵉ Chen et al., ICML 2024, "Single Focus" setting (one model per dataset — the setting that
matches ReGraph's per-dataset training). LLaGA also reports Task/Classification/General-expert
settings where a single model covers several datasets and tasks; ReGraph trains per dataset, so
Single Focus is the like-for-like row.

ReGraph reaches **86.72** against LLaGA's 88.86-89.22 and a GNN band of 87.97-89.19 — within
~2.5 points of both, with 100% legality, while emitting the class *name* as free text rather
than selecting from a label set. It comfortably beats the GPT-3.5 general-setting row (71.75).
This is the second dataset (with ogbn-arxiv) where the graph→LLM vector channel suffices,
and for the same reason: the answer is a coarse topic already implied by the centre node's text.

*Data provenance:* LLaGA's own processed release (Box) is a dead link, so Cora was reassembled
from TAPE (`xxhe/tape-cora`: text, labels, split) plus the original LINQS `cora.content` /
`cora.cites` (graph). **The join is on `cora.content` row order, not Planetoid's** — verified
100% label agreement vs 14.29% (chance) for Planetoid ordering; `tag_raw.py` re-checks this at
build time and raises on mismatch.

#### Table 5 — PubMed (LLaGA lineage, text-attributed)

Same protocol as Table 4: open-ended generation of the class name, no classification head.
60/20/20 split matching LLaGA's 6:2:2 (19,717 nodes → 11,831 / 3,943 / 3,943), 2-hop ego-subgraphs.

| Method                       | Backbone                  | PubMed (Acc)    |
| ---------------------------- | ------------------------- | --------------- |
| GCN ᵉ                       | —                        | 92.96           |
| GraphSAGE ᵉ                 | —                        | 94.87           |
| GAT ᵉ                       | —                        | 92.33           |
| SGC ᵉ                       | —                        | 87.35           |
| SAGN ᵉ                      | —                        | **95.17** |
| NodeFormer ᵉ                | —                        | 94.90           |
| GPT-3.5 (general setting) ᵉ | GPT-3.5                   | 88.00           |
| LLaGA-ND-7B ᵉ               | Vicuna-7B, frozen         | 95.03           |
| LLaGA-HO-7B ᵉ               | Vicuna-7B, frozen         | 95.03           |
| **ReGraph (ours)**     | Llama-3.1-8B-Inst, frozen | 89.98           |
| *ReGraph (ours), 3B*       | Llama-3.2-3B-Inst, frozen | *89.37*       |

ᵉ Chen et al., ICML 2024, Table 1, "Single Focus" block — the same table and setting as Table 4.

**PubMed is ReGraph's clearest loss among the text-attributed datasets: −5.19 against SAGN and
−5.05 against LLaGA**, and unlike Cora it is beaten by *every* GNN baseline except SGC. The gap is
larger than on Cora (−2.5) despite the two tasks being structurally identical, which is consistent
with PubMed having only three classes: with a 3-way decision the ceiling is high (baselines cluster
at 92-95) and a frozen vector channel that resolves topics only coarsely loses more by failing on
the residual hard cases. No intervention was attempted here — PubMed was run as a companion to Cora
and the gap was recorded, not investigated.

*Note on an earlier defect:* until 2026-08-24 this table did not exist, and the two PubMed baseline
figures quoted elsewhere in this repo (SAGN 95.17, LLaGA 95.03) carried no citation, in violation
of the sourcing rule stated at the top of this section. Both were subsequently verified against
Chen et al., Table 1 and found correct; the missing table and the missing provenance are fixed here.

#### Table 6 — NeighborhoodQA (open-ended, constructed by this repo)

No benchmark surveyed evaluates `ReGraph.md`'s motivating claim of *non-retrieval, open-ended*
graph queries, so one was built. Task: **"Which arXiv CS subject areas appear among the papers
this one cites? List every area."** The answer is a variable-length **set** of area names
generated as free text and scored by set-F1 — there is no label set to classify into. Ground
truth is exact (from ogbn-arxiv labels) and is computed on the **sampled** ego-subgraph the
model actually sees. 25,894 examples (19,902 / 1,999 / 3,993), mean 2.28 areas per answer.

| Configuration                                            | what the model can see | set-F1          | exact-set match |
| -------------------------------------------------------- | ---------------------- | --------------- | --------------- |
| No-reader control (`num_rounds=0`)                     | nothing                | 6.98            | —              |
| "Answer the centre paper's own area" (analytic shortcut) | —                     | 63.81           | —              |
| **0-hop control** (centre node only)               | centre paper's text    | 72.65           | 27.02           |
| **ReGraph** (full 2-hop)                           | centre + neighbours    | **83.87** | **45.28** |

**The neighbourhood is worth +11.2 set-F1 and +18.3 points of exact-set match.** The 0-hop
control is the load-bearing comparison: it is derived from the *identical* store with each
ego-subgraph truncated to its centre node, reusing `examples.json` byte-for-byte, so ground
truth and splits match the full run exactly and the only difference is the presence of
neighbours. Validation loss separates the same way (0.4019 vs 0.2627). The benchmark therefore
does require reading the graph, not just the queried node.

Two honest qualifications. The centre node alone already reaches 72.65, above the 63.81 analytic
shortcut — citation homophily means a paper's own abstract predicts much of its neighbourhood,
so this task rewards graph reading but does not *depend* on it the way NLGraph's structural
tasks do. And the `num_rounds=0` row (6.98) proves less than it looks: the prompt is a fixed
question, so removing the graph channel also removes any indication of which paper is being
asked about, leaving only a constant guess.

#### Table 7 — NeighborhoodQA-2hop: does ReGraph actually traverse?

Table 6's task is answerable in part from the centre node alone (citation homophily). A harder
variant isolates traversal. Question: *"Look at the papers cited by the papers this one cites.
Which subject areas appear at that second step, **excluding the ones already cited directly**?"*
The exclusion is what makes it a control: the gold set is areas at distance exactly 2 **minus every
area present at distance 1**, so a model restricted to one hop has, by construction, never seen a
single node carrying a gold label. 23,711 examples, mean 4.77 areas per answer.

The 1-hop control is built by truncating the same store — `examples.json` is **byte-for-byte
identical**, only the graphs shrink (53.87 → 8.43 mean nodes), so splits, answers and question text
match exactly.

| Configuration                                                | mean nodes seen | set-F1          | exact-set match |
| ------------------------------------------------------------ | --------------- | --------------- | --------------- |
| **1-hop control** (gold labels structurally invisible) | 8.43            | 48.66           | 1.42            |
| **ReGraph** (full 2-hop)                               | 53.87           | **56.68** | **2.86**  |

Paired over the 3,882 shared test examples the difference is **+7.95 set-F1, SEM 0.33 (24 SE)** —
real, and not attributable to noise. Two things follow, one positive and one not.

**ReGraph does extract genuine two-hop information.** A model that cannot see any gold-labelled
node scores 48.66, and ReGraph clears it by 8 points on identical examples.

**But the reader's diffusion operator is not what does it.** The per-round hop distribution on this
task — the α of `ReGraph.md` §2.3, on a benchmark explicitly designed to require hop 2 — is

```
round 0: [1.000000, 0.000000, 0.000000]
round 1: [1.000000, 0.000000, 0.000000]
round 2: [0.999996, 0.000000, 0.000000]
```

α collapses onto hop 0 completely. The two-hop signal therefore arrives through the **graph
encoder**, whose 4 TransformerConv layers already propagate 4 hops before the reader sees the node
memory, and on this task §2.3's diffusion is redundant with it rather than additive. That it
reproduces on a benchmark built to defeat exactly that shortcut makes it the strongest version of
the finding **for the vector-only configuration**.

**It does not generalise to the token-channel configuration** — a claim made in an earlier revision
of this section and withdrawn here. The full-scale SceneGraphs dual-channel run
(`runs/scene_graphs/dual-full`, §"The one intervention that worked") shows α spread genuinely
across hops at test time, averaged over all 20,025 examples:

```
round 0: [0.5348, 0.1224, 0.3428]
round 1: [0.4909, 0.3632, 0.1459]
round 2: [0.4404, 0.4399, 0.1197]
```

So the honest statement is conditional: **α collapses onto hop 0 whenever the reader is the only
graph channel, and spreads once a token channel is also present.** Two readings are still open —
it may be the token channel that changes what the reader is asked to supply, or simply that
SceneGraphs rewards multi-hop reading where NeighborhoodQA-2hop does not. Distinguishing them needs
a vector-only SceneGraphs run compared against this one at equal budget, which has not been done.
Until then, "§2.3 can be removed" is supported **only** for the vector-only setting. Recorded in
[docs/OPEN-QUESTIONS.md](docs/OPEN-QUESTIONS.md).

**Also note the floor.** 48.66 without any gold-labelled node visible means arXiv area
co-occurrence is highly predictable — the benchmark is harder than Table 6 but still not a pure
traversal test. A reader should treat +7.95 over the control, not 56.68 in isolation, as the
measure of two-hop reading.

#### Table 8 — StructuralAnomaly (synthetic, anchor-free, constructed by this repo)

`ReGraph.md` §2.1 leads with an **anchor-free** query — *"Which region of the graph is becoming
structurally unstable?"*, where "all nodes receive the `none` marker" and the model "locates
relevant graph regions through semantic and structural attention rather than beginning from a
predefined question entity". **Every other benchmark in this repo is anchored** (GraphQA marks
question entities, the TAG datasets and NeighborhoodQA mark a centre node, NLGraph names both
endpoints), so the specification's headline capability had never been tested. This synthetic
benchmark tests it.

Each graph holds 5 equally sized thematic communities (12 nodes each); exactly one is wired far
more densely (`p_hot` 0.60 vs `p_in` 0.12). The question names no node. The answer is the dense
community's theme, as free text. 11,000 examples per arm.

| Configuration | Llama-3.1-8B | Llama-3.2-3B | Legality |
| --- | --- | --- | --- |
| Chance — **analytic, not estimated** | 20.00 | 20.00 | — |
| **Control arm** (density contrast removed, label random) | **19.80** | **20.05** | 100.00 |
| **ReGraph** | **99.05** | **99.15** | 100.00 |

**The control is the point.** All five themes appear in every graph, each is equally likely to be
the dense one, and communities are equal in size, so no topology-ignoring policy can beat 1/k.
The control arm removes the density contrast and randomises the label, making the task
unanswerable by construction — any score above chance there would mean the generator leaks and
99.05 is void. It lands at 19.80 on the 8B backbone and 20.05 on the 3B one — **0.2 SE and 0.06 SE from
analytic chance**. The leakage check therefore passes *independently on two backbones*, so it is
not an 8B coincidence. This is the only benchmark here with a falsifiable validity check.

**Both channels are load-bearing.** Topology finds the dense region but cannot name it; text
names all five themes but cannot say which is anomalous. That is exactly the `Read` operation of
§2.3, and it is what NeighborhoodQA (partly solvable from the centre node's text, 72.65 of 83.87)
and NLGraph (pure topology, no semantics) each fail to isolate.

**It measures the graph interface, not the language model.** Halving the backbone moves the
result by +0.10 (99.05 → 99.15, well inside noise at n=2,000), and the 3B run reaches a *lower*
validation loss (0.0089 vs 0.0143). This is the same signature as NeighborhoodQA (−0.05) and the
opposite of WebQSP (−11.06), which is how that dataset was diagnosed as scoring on the backbone's
parametric knowledge rather than on graph reading.

**Diagnostics.** α collapses onto **hop 2** (`[0.000, 0.000, 0.999]` in all three rounds) with
gates at 0.62 / 0.83 / 0.82. On the control arm the gates collapse to 0.05 / 0.10 / 0.11 — the
model learns to shut a channel carrying nothing, which is an independent sanity signal.

**Two limits, stated plainly.** The structural cue is **degree** (+4.4 edges for dense-community
nodes), a first-order local quantity one message-passing round computes; this is anchor-free
localization by a *local* cue, not path-level reasoning — consistent with ReGraph remaining at
chance on NLGraph. And at `density_ratio = 5.0` the result is **saturated**, so the informative
quantity is the accuracy-vs-`density_ratio` curve and the threshold where a method falls back to
chance, not this single point. Full documentation, data and a standalone scorer:
[benchmarks/structuralanomaly/](benchmarks/structuralanomaly/).

### Backbone sensitivity: Llama-3.1-8B-Instruct vs Llama-3.2-3B-Instruct

Same code, same preprocessed data, same hyperparameters — only `llm.name`, `llm.d_llm` (4096 →
3072) and `llm.num_layers` (32 → 28) change. The default everywhere else in this README remains
Llama-3.1-8B-Instruct. Trainable parameters: 32.70M → 30.56M.

| Dataset              | 8B (default) | 3B    | Δ     |
| -------------------- | ------------ | ----- | ------ |
| ExplaGraphs          | 92.42        | 90.43 | -1.99  |
| SceneGraphs          | 51.83        | 53.24 | +1.41  |
| WebQSP               | 62.22        | 51.17 | -11.06 |
| ogbn-arxiv           | 71.75        | 72.28 | +0.53  |
| Cora                 | 86.72        | 88.56 | +1.85  |
| PubMed               | 89.98        | 89.37 | -0.61  |
| ogbn-products        | 74.21        | 74.27 | +0.06  |
| NLGraph connectivity | 49.33        | 54.18 | +4.85  |
| NLGraph cycle        | 52.88        | 47.12 | -5.76  |
| NeighborhoodQA       | 83.87        | 83.83 | -0.05  |
| StructuralAnomaly    | 99.05        | 99.15 | +0.10  |
| — its control arm   | 19.80        | 20.05 | +0.25  |

**Backbone size barely matters — except where the answer is an entity name.** Eight datasets move
by under 2 points, two of them *upward* with the smaller model, and both NLGraph tasks stay at
chance regardless. WebQSP is the outlier at −11.1 (6.3 SE).

That asymmetry is diagnostic. A channel-limited system should be insensitive to LLM capacity:
if the bottleneck is what the graph→LLM interface can carry, spare language-model capacity buys
nothing. Every dataset where ReGraph's score reflects graph reading behaves that way. WebQSP
does not — and it is precisely the dataset where the reader was measured not to localize the
answer (gold entity at median rank 301/1371), implying its score came from the backbone's
parametric knowledge of famous entities. Halving the backbone removes 11 points of exactly that.

**NeighborhoodQA passes the same test** (83.87 → 83.83): the benchmark constructed here measures
the graph interface rather than language-model capacity, which is what a graph-reasoning
benchmark should do.

On ExplaGraphs the 3B result is also the first *same-backbone* comparison with GRAFF, whose main
table uses Llama-3.2-3B (ExplaGraphs graphs average 5.17 nodes, so no retrieval step is involved
and the setups line up):

| Method (3B-class backbone)                | ExplaGraphs     |
| ----------------------------------------- | --------------- |
| G-Retriever                               | 83.7            |
| LoRA                                      | 88.9            |
| **ReGraph (ours), no text channel** | **90.43** |
| GRAFF                                     | 92.5            |

#### Reading the tables

| Dataset        | vs best published                                                                      | What the answer requires                        |
| -------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------- |
| ExplaGraphs    | **−0.08** vs GRAFF (92.42 vs 92.5, tied); **+5.4** vs G-Retriever w/ LoRA | nothing from the graph (question text suffices) |
| ogbn-arxiv     | +43.3 vs GraphTranslator (different protocol)                                          | coarse topic of one node                        |
| Cora           | −2.5 vs LLaGA / GNN band                                                              | coarse topic of one node                        |
| PubMed         | **−5.05** vs LLaGA, −5.19 vs SAGN (Table 5)                                    | coarse topic of one node                        |
| NeighborhoodQA | +20.1 over its shortcut floor                                                          | a*set* of areas aggregated over neighbours    |
| WebQSP         | −9.98 vs GRAFF; −11.6 vs G-Retriever w/ LoRA                                         | an exact entity surface form                    |
| SceneGraphs    | −38.4 vs GRAFF                                                                        | binding one attribute to one object             |
| NLGraph        | at chance                                                                              | a property computed from topology               |

One mechanism orders all five: the vector channel relays **semantics already present in node
features**, but cannot **bind attributes to objects** or **compute relations**. Adding a token
channel supplies exactly those two capabilities, which is why row 15 jumps SceneGraphs from
51.83 to **89.41** — and at matched budget the reading rounds still contribute +7.00 pp on top of
serialization.

### Benchmark 1 — GraphQA (G-Retriever, NeurIPS 2024)

| Method                        | Backbone                      | ExplaGraphs (Acc) | SceneGraphs (Acc)       | WebQSP (Hit@1)          |
| ----------------------------- | ----------------------------- | ----------------- | ----------------------- | ----------------------- |
| Zero-shot¹                   | Llama2-7b                     | 56.50             | 39.74                   | 41.06                   |
| Prompt tuning¹               | Llama2-7b, frozen             | 57.63 ± 2.43     | 63.41 ± 0.24           | 48.34 ± 0.64           |
| GraphToken¹                  | Llama2-7b, frozen             | 85.08 ± 5.51     | 49.03 ± 1.05           | 57.05 ± 0.74           |
| G-Retriever¹                 | Llama2-7b, frozen             | 85.16 ± 0.92     | **81.31 ± 1.62** | **70.49 ± 1.21** |
| G-Retriever w/ LoRA¹         | Llama2-7b, tuned              | 87.05 ± 3.29     | **86.83 ± 0.72** | **73.79 ± 0.70** |
| **ReGraph (this repo)** | Llama-3.1-8B-Instruct, frozen | **92.42**² | 51.83                   | 62.22                   |

¹ He et al., 2024 (arXiv:2402.07630) Table 3.
² Corrected run (`runs/expla_graphs/fix-seed0`); the pre-fix three-seed spread was 92.42 ± 0.36.

ReGraph wins on ExplaGraphs and loses clearly on the other two. Note the backbone differs in
ReGraph's favour, so the two deficits are if anything understated.

### Benchmark 2 — ogbn-arxiv (GraphTranslator, WWW 2024)

Chosen because GraphTranslator shares ReGraph's two hard constraints — frozen LLM, graph never
serialized — so it compares the same design choice rather than a text channel.

| Method                          | Backbone                      | Protocol             | Top-1           | Top-3           | Top-5           |
| ------------------------------- | ----------------------------- | -------------------- | --------------- | --------------- | --------------- |
| Majority class                  | —                            | —                   | 21.90           | —              | —              |
| GraphTranslator³               | ChatGLM2-6B, frozen           | **zero-shot**  | 28.48           | 37.62           | 39.87           |
| **ReGraph (this repo)**   | Llama-3.1-8B-Instruct, frozen | **supervised** | **71.75** | **92.58** | **96.40** |
| ReGraph + alignment pretraining | same                          | supervised           | 71.45           | 92.28           | 96.65           |

³ Zhang et al., WWW 2024, Table 1.

**This is not a like-for-like win.** GraphTranslator is zero-shot; ReGraph is trained on 20,000
ogbn-arxiv train nodes with a different and stronger backbone. 71.75 is roughly where a
well-tuned supervised GNN lands on ogbn-arxiv. What the number does establish is that ReGraph
behaves sensibly here: 100% legality, 37 of 40 categories used, against a 21.90 floor.
Evaluated on GraphTranslator's exact 4,000-node test subset. Each example is a sampled 2-hop
ego-subgraph (fanout 10, mean 45.5 nodes), not the full 169k-node graph. Top-3/Top-5 come from
likelihood ranking over the 40 category names (`regraph.eval_rank`), since ReGraph has no
classification head; that ranking reproduces Top-1 = 71.60 against the generated 71.75, which
is the consistency check for the metric. Length-normalized ranking gives 71.23 / 90.13 / 94.60.

*GraphTranslator's second dataset, Taobao, is not public* — 980k Alibaba users with behavioural
data, and its open-ended tasks are scored by human/ChatGPT rating. Neither the data nor the
metric is reproducible, so only the ArXiv half is covered.

### Benchmark 3 — NLGraph (Wang et al., NeurIPS 2023)

Structural graph reasoning. This is the benchmark family `ReGraph.md` §2.1 draws its own
example from, and the one that tests the paper's opening claim that GraphRAG "can not handle
non-retrieval query" — there is no span to retrieve, only topology to reason over. Nodes are
bare integers with no attributes, so every node shares one placeholder embedding and is
distinguishable only by topology and its role marker.

Baselines are text-davinci-003 with the edge list **in the prompt** (their Table 2); ReGraph
strips the edge list and reads topology through the encoder. Their protocol is zero-/few-shot
prompting, ours is supervised on the train split — an asymmetry that favours ReGraph.

| Method                        | Connectivity (E/M/H/Avg)               | Cycle (E/M/H/Avg)                      |
| ----------------------------- | -------------------------------------- | -------------------------------------- |
| Random                        | 50.00 / 50.00 / 50.00 /**50.00** | 50.00 / 50.00 / 50.00 /**50.00** |
| Zero-shot⁴                   | 83.81 / 72.75 / 63.38 / 71.31          | 50.00 / 50.00 / 50.00 / 50.00          |
| Few-shot⁴                    | 93.75 / 83.83 / 76.61 / 84.73          | 80.00 / 70.00 / 61.00 /**70.33** |
| CoT⁴                         | 94.32 / 82.17 / 77.21 / 84.57          | 84.67 / 63.33 / 53.25 / 66.75          |
| CoT+SC⁴                      | 93.18 / 84.50 / 82.79 /**86.82** | 82.00 / 63.67 / 53.50 / 66.39          |
| **ReGraph (this repo)** | 66.07 / 47.96 / 43.70 /**49.33** | 64.00 / 49.53 / 54.24 /**52.88** |

⁴ Wang et al., NeurIPS 2023, Table 2 (text-davinci-003, standard set).

**ReGraph is at chance on both tasks.** Connectivity 49.33 against a 50.00 random baseline
(n=371, SE 2.6 pp) and cycle 52.88 (n=191, SE 3.6 pp, 0.8 SE above random). The only subsets
showing signal are the easiest — connectivity easy 66.07 (n=56, 2.4 SE above random) and cycle
easy 64.00 (n=25, not significant) — i.e. the smallest graphs. Validation loss confirms it
independently: both runs converged to 0.3471 / 0.3474 against a chance floor of
`ln(2)/2 = 0.3466`, and the fusion gates fell steadily during training (connectivity ended at
0.09 / 0.20 / 0.22), meaning the model progressively learned to ignore a graph channel that
was telling it nothing.

The data is not at fault: parsed graphs reproduce NLGraph's ground-truth labels on 400/400
sampled examples for both tasks, verified independently with union-find.

**This is the most consequential negative result in the project.** ReGraph's stated motivation
is the non-retrieval structural query, and on the purest such benchmark it fails while
text-serialized prompting succeeds. It also completes the mechanism picture: the graph→LLM
channel can relay semantics that are already present in node features (arxiv 71.75, where the
sbert embedding of title+abstract nearly contains the answer), but it cannot convey a property
that has to be *computed from topology* — which is what "graph reading" would have to mean.

Caveat worth testing before treating this as final: `reader.max_hops` is 2, inherited from the
GraphQA protocol, while connectivity on 20-35 node graphs may need longer reach. K is a config
value (`reader.max_hops=4`), not a code change.

### Comparison with GRAFF (Findings of EACL 2026)

GRAFF (Chaudhary et al.) evaluates on **exactly the three GraphQA datasets this repo already
covers**, plus a synthetic variant. Its architecture is the closest published relative of
ReGraph: a GAT over node embeddings injected into an **intermediate decoder layer** via a
residual add — the same mid-stack fusion idea, at one layer instead of interleaved over T rounds.

| Method (LLaMA-3.2-3B)                          | WebQSP         | ExplaGraph     | SceneGraph     | SyntheticGraph |
| ---------------------------------------------- | -------------- | -------------- | -------------- | -------------- |
| G-Retriever⁵                                  | 67.4           | 83.7           | 82.3           | 56.4           |
| LoRA⁵                                         | 71.1           | 88.9           | 85.3           | 58.3           |
| **GRAFF**⁵                              | **72.2** | **92.5** | **90.2** | **79.4** |
| *ReGraph (this repo, Llama-3.1-8B-Instruct)* | *62.22*      | *92.42*      | *51.83*      | —             |

⁵ Chaudhary et al., Findings of EACL 2026, Table 1.

**The ReGraph row is italic because it is NOT comparable, for two reasons.**

1. **Retrieved vs full graphs.** GRAFF's Table 2 reports WebQSP at **8.39 average nodes** and
   SceneGraph at 8.21 — they evaluate on G-Retriever's PCST-*retrieved* subgraphs. ReGraph as
   specified reads the full graph (WebQSP ~1,371 nodes; `ReGraph.md` §3.2 has no retrieval
   step). Their limitations section is explicit: "we demonstrate the effectiveness of GRAFF on
   small-sized sub-graphs retrieved from a much larger graph."
2. **Backbone.** LLaMA-3.2-3B (and Qwen-2.5-3B) against our Llama-3.1-8B-Instruct.

**Attempt to close gap 1, and why it failed.** `regraph.data.pcst_retrieve` ports
G-Retriever's `retrieval_via_pcst` and was verified to produce *identical* node and edge counts
to the original on sampled WebQSP graphs. With G-Retriever's own WebQSP settings
(`topk=3, topk_e=5, cost_e=0.5`) it yields **2.54 nodes per graph**, not 8.39, and the gold
answer entity survives retrieval in only **13-15%** of examples versus **95%** in the full
graph. Sweeping the budget does not help — node count stays ~2.5 while recall *falls* as topk
rises. A subgraph that contains the answer 13% of the time cannot support a 70+ Hit@1, so the
retrieved-variant comparison was abandoned rather than trained on. GRAFF's own repository is an
empty stub ("code will be released shortly"), so their exact retrieval could not be recovered.

**SyntheticGraph is not reproducible either**: it is described only as shuffling nodes and
arguments of ExplaGraphs while preserving edges, and neither the data nor the construction code
is released. It is, however, the most interesting of their four datasets for ReGraph, because it
is designed so a model *must* use the graph — which is precisely what our ExplaGraphs diagnostic
says ReGraph does not do there (evidence vector 98% example-invariant). GRAFF scores 79.4 on it
against LoRA's 58.3; ReGraph would likely score near the text-only baseline.

**One transferable idea.** GRAFF avoids the failure this project spent the most time on. It
reuses *the LLM's own embeddings* for node features rather than an external encoder, explicitly
"eliminating embedding space misalignment". Our measurements say ReGraph's `W_O` never learns a
usable sbert→LLM map from answer likelihood alone, and that alignment pretraining does not fix
it. Encoding nodes in the LLM's own embedding space would sidestep the problem rather than try
to learn around it.

### Attempts to improve the two losing datasets — all negative

Every lever inside ReGraph's specified architecture was tested against matched-step controls on
SceneGraphs. None beat the protocol (SE of a difference ≈ 1.8 pp):

| Lever                                              | Result                                                                  |
| -------------------------------------------------- | ----------------------------------------------------------------------- |
| Learning rate 1e-5 → 3e-4                         | no gain; drives the fusion gate to ~0                                   |
| Per-module `lr_mult`                             | no gain (AdamW normalizes per-parameter step size)                      |
| `W_O` init zeros → normal                       | +1.46 pp, 0.81 SE — not significant                                    |
| Query tokens N_B 8 → 32 → 64                     | 43.67 / 43.47 / 43.00 — flat                                           |
| Diffusion depth K                                  | flat                                                                    |
| **Alignment pretraining** (arXiv)            | val loss 0.2783 → 0.2582 but test 71.75 → 71.45 —**no effect** |
| LLM-embedding node features + coords (SceneGraphs) | 36.73 vs 43.67 control —**−6.94 pp, significantly worse**       |
| sbert + numeric box coordinates (SceneGraphs)      | 44.80 vs 43.67 at 10k;**full run 51.50 vs 51.83** — no effect    |

Two genuine bugs *were* found and fixed: the evidence `R` was dropped out twice against the
spec's single `Dropout(R)`, and scientific-notation CLI overrides silently parsed as strings.
Both are correctness fixes with no measurable effect on scores.

### The one intervention that worked: adding a token channel alongside the reader

Every attempt above tried to improve the graph→LLM *vector* channel and failed, because that
channel cannot bind attributes to objects or compute relations. This attempt instead **adds a
second channel** and keeps the first: the graph is *also* serialized into the prompt as CSV
text (G-Retriever's `desc` format) while all three Read-Fuse-Replace rounds stay active.

**The architecture is unchanged.** Diffing the resolved configs, 5 of 66 values differ and none
of them is structural — `num_query_tokens=8`, `num_rounds=3`, `max_hops=2`, 8 reader heads,
4-layer graph transformer, shared Reader/Fuse, frozen Llama-3.1-8B-Instruct, sbert attributes,
lr 1e-5 are all exactly as specified. The only substantive change is `data.serialize_graph`;
the rest are a longer prompt budget and a smaller eval batch to fit the longer sequences.

The input sequence goes from `[T_q ; B_base ; boundary ; answer]` (~40 tokens) to
`[graph CSV ; T_q ; B_base ; boundary ; answer]` (~1,500 tokens), so the graph now reaches the
LLM by two routes: as **tokens** its own attention reads at full bandwidth and can copy strings
from, and as **vectors** through the unchanged encoder → reader → gated fusion path written into
the B-token positions at layers 8/16/24.

| SceneGraphs arm (3,000 steps, 1,500 eval examples) | accuracy        | train loss |
| -------------------------------------------------- | --------------- | ---------- |
| vector channel only (= ReGraph as specified)       | 34.93           | 0.9182     |
| token channel only (`num_rounds=0`)              | 74.93           | 0.5129     |
| **both channels**                            | **81.93** | 0.3229     |

**Iterative graph reading adds +7.00 pp on top of serialization (4.6 SE, highly significant).**
It is not subsumed by the token channel. This is the first significant positive intervention in
the project, and it contradicts the prediction made before the run — which is why the
`num_rounds=0` control was run at all.

**Trained to convergence, the both-channels arm reaches 89.41** on the full 20,025-example test set
(6 epochs, best validation loss 0.2239 at epoch 3; `runs/scene_graphs/dual-full`). That clears
G-Retriever's frozen 81.31 by 8.10 pp, **G-Retriever w/ LoRA's 86.83 by 2.58 pp (11.9 SE)** and
LoRA-3B's 85.3 by 4.11 — while keeping the LLM frozen, against a LoRA-tuned competitor. GRAFF
(90.2) remains 0.79 ahead (3.6 SE). Two asymmetries to keep in view: the full graph is serialized
(truncated to 1,536 tokens) rather than a PCST subgraph, and the backbone is 8B against their
7B/3B. The no-reader control has not been rerun at this budget, so the +7.00 pp reader
contribution above is still the 3,000-step measurement, not a measurement at 89.41.

**This changes what ReGraph is being claimed to be.** `ReGraph.md` §3.2 keeps the graph out of
the context, and that premise is the paper's stated advantage over GraphRAG. A model with a
token channel is no longer ReGraph-as-specified; it is **ReGraph as an augmentation to a
GraphRAG-style pipeline**. Reported on its own row, never mixed with the protocol numbers
(docs/OPEN-QUESTIONS.md Q23).

Two implementation details that are easy to get wrong, both measured:

* **Do not use PCST retrieval here.** On SceneGraphs it keeps the gold answer string in only
  37.3% of examples versus 81.7% for the full graph, which would cap the ceiling below the
  existing 51.83 baseline.
* **Budget the graph text, not the question.** The graph is prepended, so truncating the whole
  prompt silently drops the question on long graphs (p90 ≈ 1,700 tokens), leaving the model
  answering blind on ~10-15% of examples. Caught by decoding an actual training prompt before
  trusting the flag.

**Tested on four datasets — and the direction is not consistent.**

| dataset              | reader alone   | A: text + reader | B: text only | A − B            |         |
| -------------------- | -------------- | ---------------- | ------------ | ----------------- | ------- |
| SceneGraphs †       | 51.83          | 81.93            | 74.93        | **+7.00**   | 4.7σ   |
| ExplaGraphs          | 92.42          | 86.82            | 81.05        | **+5.77**   | 2.6σ   |
| NLGraph connectivity | 49.33 (chance) | 87.87            | 75.74        | **+12.13**  | 4.3σ   |
| NLGraph cycle        | 52.88 (chance) | 75.92            | 90.58        | **−14.66** | −3.9σ |

All four are significant; three positive, one negative. **Two generalizations were made and
both were falsified by the next dataset**, and both are recorded here rather than quietly
dropped:

1. After SceneGraphs and ExplaGraphs, "iterative reading always adds on top of serialization" —
   falsified by NLGraph cycle (−14.66).
2. Then "the reader helps iff it has standalone signal" (cycle's reader is at chance, so noise
   injection would explain the loss) — falsified by NLGraph connectivity, whose reader is
   *also* at chance yet gains +12.13.

**The honest current conclusion is narrower: adding a token channel to ReGraph produces a large
change on every dataset tested, but the sign is dataset-dependent and the present evidence does
not predict it.** One unverified observation that may explain the NLGraph split: connectivity
supplies explicit `source`/`target` role markers (the only use of `ROLE_TARGET` in the project),
giving diffusion an anchor to propagate from, whereas cycle is anchor-free and the reader has no
starting point. Testing that needs a purpose-built experiment, not more pattern-matching on
these four points.

**But the token channel is not a free win.** On ExplaGraphs it *loses* 11.4 pp on its own
(81.05 vs the 92.42 protocol result) and the full augmentation still trails the pure protocol
model (86.82 vs 92.42, 3.07 SE). That task is solvable from the question text — the graph
evidence there is 98% example-invariant — so serializing it only adds distraction. The
augmentation pays off where the graph is load-bearing (SceneGraphs) and costs where it is not.

*Status: a full-scale SceneGraphs run is training (20.4 h/epoch at ~1,500-token sequences); its
row above is the 3,000-step comparison on a 1,500-example subset.*

### Why the results split the way they do

One mechanism explains all four datasets. The graph→LLM channel is a convex average of node
vectors injected mid-stack; it carries **coarse semantics** but not **identity or surface form**.

| Dataset         | What the answer requires             | Result                                                                                    |
| --------------- | ------------------------------------ | ----------------------------------------------------------------------------------------- |
| ExplaGraphs     | nothing from the graph               | 92.42 — but the evidence vector is 98% example-invariant, so this is backbone, not graph |
| **arXiv** | coarse topic of one node (~5.3 bits) | **71.75**                                                                           |
| WebQSP          | an exact entity surface form         | 62.22 (single-answer 51.7 vs multi-answer 73.2)                                           |
| SceneGraphs     | bind one attribute to one object     | 51.83 (colour questions 28.1)                                                             |

#### SceneGraphs: two further attempts, both failed (2026-08-22)

Targeted at the two measured causes rather than at hyperparameters, which were already exhausted.

1. **Node features in the LLM's own embedding space** (GRAFF Eq. 4: `c_v` = mean of the frozen
   Llama input embeddings over the node's text tokens), replacing sbert. Result: **36.73 vs
   43.67**, significantly *worse*. The transfer was mistaken: in GRAFF node embeddings become
   **tokens in the LLM input sequence**, so LLM-space features genuinely remove a misalignment;
   in ReGraph they pass through a 4-layer graph transformer and `W_O`, which remap them anyway,
   so nothing is aligned — while mean-pooled token embeddings are a much weaker sentence
   representation than sbert (train loss 1.0497 vs 0.9568).
2. **Numeric box coordinates** appended to the sbert features (x, y, w, h, area, cx, cy; boxes
   parsed for 99.96% of 833,373 texts). Result: **44.80 vs 43.67**, +1.13 pp at 0.62 SE — not
   significant, despite spatial questions being the largest category (6,916/20,025) at 50.4%.

Attempt 2 was then run at full scale, and its per-category breakdown is the single clearest
result in this project. Exact bounding boxes were handed to the model as explicit numeric
features; the category they exist to fix moved by **0.61 pp**:

| category             | n                | baseline        | + coordinates   | delta                      |
| -------------------- | ---------------- | --------------- | --------------- | -------------------------- |
| spatial / relational | 6,916            | 50.94           | 51.55           | **+0.61**            |
| existence (yes/no)   | 5,304            | 55.34           | 53.09           | −2.24                     |
| colour               | 957              | 29.26           | 26.33           | −2.93                     |
| object identity      | 4,655            | 50.57           | 50.10           | −0.47                     |
| other                | 2,193            | 58.69           | 61.42           | +2.74                      |
| **ALL**        | **20,025** | **51.83** | **51.50** | **−0.33** (0.66 SE) |

Giving the model the exact coordinates does not help, because "which side is the pillow on"
requires *comparing* two objects' coordinates — a relational computation, not a lookup, and
NLGraph already showed ReGraph is at chance on computed properties. Spatial and yes/no
existence questions fail for the same reason: both need a property **computed over** the graph
rather than **read from** it.

**The information was never the bottleneck; the computation is.** That is why eight successive
interventions — learning rate, `lr_mult`, `W_O` init, diffusion depth K, query tokens at 8/32/64,
alignment pretraining, LLM-space features, and numeric coordinates — all failed to move
SceneGraphs. Closing the gap to G-Retriever's 81.31 requires a change to the method (letting
node text reach the token stream, or injecting evidence as tokens rather than as a mid-stack
residual), not a better configuration of this one.

**Follow-up that resolves this.** The limitation above is about the *vector* channel, and it is
removed by adding a token channel rather than by improving the vector one — see "The one
intervention that worked" above: 34.93 → 81.93 at matched budget on SceneGraphs (89.41 trained to
convergence), of which +7.00 pp is attributable to the reading rounds themselves.

**Diagnostic caveat:** fusion-gate means are *not* a health signal on their own, contrary to
`docs/components/05-fuse.md` §5.1. The aligned arXiv model runs gates at 0.06 with ‖R‖ = 654,
giving *more* effective injection (‖g·R‖/‖B_pre‖ = 659%) than the control's gate 0.67 with
‖R‖ = 11.3 (52%). Always report ‖R‖ alongside the gate.

## Reproducing the reported results

Every number in this README comes from a checkpoint in `runs/`. Two paths are given per
dataset: **(a)** re-evaluate our checkpoint (minutes), or **(b)** rebuild from raw data and
retrain (hours). Both need the environment set up once:

```bash
pip install -r requirements.txt && pip install -e .
source scripts/env.sh                 # HF_HOME -> /mnt/ssd1/zhuowei/hf-cache
hf auth login                         # Llama-3.1-8B-Instruct is a gated repo
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
```

Checkpoints store **only the ~32.7M trainable parameters**, not the frozen LLM, so the backbone
above must be present to load any of them.

### (a) Re-evaluate a released checkpoint

```bash
python -m regraph.eval --config configs/<CONFIG>.yaml --ckpt runs/<CKPT>/best.pt --split test
```

| Dataset                    | Reported        | Metric | `<CONFIG>`                  | `<CKPT>`                          |
| -------------------------- | --------------- | ------ | ----------------------------- | ----------------------------------- |
| ExplaGraphs                | **92.42** | Acc    | `expla_graphs`              | `expla_graphs/fix-seed0`          |
| SceneGraphs                | **51.83** | Acc    | `scene_graphs`              | `scene_graphs/fix-seed0`          |
| WebQSP                     | **62.22** | Hit@1  | `webqsp`                    | `webqsp/fix-seed0`                |
| ogbn-arxiv                 | **71.75** | Top-1  | `arxiv`                     | `arxiv/control`                   |
| Cora                       | **86.72** | Acc    | `cora`                      | `cora/seed0`                      |
| PubMed                     | **89.98** | Acc    | `pubmed`                    | `pubmed/seed0`                    |
| ogbn-products (subset)     | **74.21** | Acc    | `products`                  | `products/seed0`                  |
| NLGraph connectivity       | **49.33** | Acc    | `nlgraph_connectivity`      | `nlgraph_connectivity/seed0`      |
| NLGraph cycle              | **52.88** | Acc    | `nlgraph_cycle`             | `nlgraph_cycle/seed0`             |
| NeighborhoodQA             | **83.87** | set-F1 | `arxiv_nbrqa`               | `arxiv_nbrqa/seed0`               |
| — its no-reader control   | 6.98            | set-F1 | `arxiv_nbrqa`               | `arxiv_nbrqa/noreader`            |
| — its 0-hop control        | 72.65           | set-F1 | `arxiv_nbrqa_zerohop`       | `arxiv_nbrqa_zerohop/seed0`       |
| NeighborhoodQA-2hop        | **56.68** | set-F1 | `arxiv_nbrqa_hop2`          | `arxiv_nbrqa_hop2/seed0`          |
| — its 1-hop control        | 48.66           | set-F1 | `arxiv_nbrqa_hop2_1hop`     | `arxiv_nbrqa_hop2_1hop/seed0`     |
| StructuralAnomaly          | **99.05** | Acc    | `synth_anomaly`             | `synth_anomaly/seed0`             |
| — its unanswerable control | 19.80           | Acc    | `synth_anomaly_control`     | `synth_anomaly_control/seed0`     |
| StructuralAnomaly, 3B      | **99.15** | Acc    | `synth_anomaly` ᵍ           | `synth_anomaly/llama3b`           |
| — its control, 3B          | 20.05           | Acc    | `synth_anomaly_control` ᵍ   | `synth_anomaly_control/llama3b`   |
| SceneGraphs + token channel | **89.41** | Acc    | `scene_graphs_dual`         | `scene_graphs/dual-full`          |
| NLGraph conn. + token ch.  | 87.87 / 75.74   | Acc    | `nlgraph_connectivity_dual` | `.../dual-A`, `.../dual-B`      |
| NLGraph cycle + token ch.  | 75.92 / 90.58   | Acc    | `nlgraph_cycle_dual`        | `.../dual-A`, `.../dual-B`      |
| ExplaGraphs + token ch.    | 86.82 / 81.05   | Acc    | `expla_graphs_dual`         | `expla_graphs/dual-A`, `dual-B` |
| SceneGraphs + coordinates  | 51.50           | Acc    | `scene_graphs_coords`       | `scene_graphs_sbert_coords/full`  |
| arXiv + alignment pretrain | 71.45           | Top-1  | `arxiv`                     | `arxiv/aligned`                   |

ᵍ Rows marked ᵍ are 3B runs: append the backbone overrides to the command, exactly as they were
trained —
`llm.name=meta-llama/Llama-3.2-3B-Instruct llm.d_llm=3072 llm.num_layers=28`.
There is no separate 3B config; every 3B result in this repo is the base config plus these three
overrides.

arXiv Top-3/Top-5 come from a separate ranking pass:

```bash
python -m regraph.eval_rank --config configs/arxiv.yaml --ckpt runs/arxiv/control/best.pt
```

Per-difficulty NLGraph tables (matching the paper's layout): `python scripts/nlgraph_report.py`.

### (b) Rebuild from raw data

Data acquisition per dataset is in [scripts/DATASETS.md](scripts/DATASETS.md). Once the raw
files are in place, each dataset follows the same three steps:

```bash
# GraphQA — needs G-Retriever's train_dev.tsv / questions.csv / sceneGraphs.zip under data/raw/
python -m regraph.data.preprocess       --config configs/expla_graphs.yaml
python -m regraph.train                 --config configs/expla_graphs.yaml run_name=mine
python -m regraph.eval                  --config configs/expla_graphs.yaml \
    --ckpt runs/expla_graphs/mine/best.pt --split test          # also: scene_graphs, webqsp

# ogbn-arxiv — OGB download + titleabs.tsv + GraphTranslator's 4,000-node test subset
python -m regraph.data.preprocess_arxiv --config configs/arxiv.yaml
python -m regraph.train                 --config configs/arxiv.yaml run_name=mine

# Cora / PubMed / ogbn-products — TAPE sources, see DATASETS.md §4
python -m regraph.data.preprocess_tag   --config configs/cora.yaml       # pubmed, products
python -m regraph.train                 --config configs/cora.yaml run_name=mine

# NLGraph — clone github.com/Arthur-Heng/NLGraph, point data.nlgraph_raw_dir at it
python -m regraph.data.preprocess_nlgraph --config configs/nlgraph_cycle.yaml
python -m regraph.train                   --config configs/nlgraph_cycle.yaml run_name=mine

# NeighborhoodQA — generated from ogbn-arxiv, no extra download beyond the arXiv raw files
python -m regraph.data.preprocess_nbrqa --config configs/arxiv_nbrqa.yaml
python -m regraph.train                 --config configs/arxiv_nbrqa.yaml run_name=mine
```

Preprocessing self-checks against `docs/experimental-protocol.md` and **refuses to continue** on
a split-size or graph-statistics mismatch, so a wrong raw file fails loudly rather than silently
producing different numbers.

### What is *not* reproducible from a checkpoint

The hyperparameter sweeps and feature-engineering runs used a standalone harness
(`runs/_logs/sweep.py`) that reports metrics without saving models. Affected: the SceneGraphs
sweep (rows A–I), the LLM-embedding and coordinate feature runs, the learning-rate probes, and
the 3,000-step SceneGraphs dual-channel A/B (81.93 / 74.93). Their raw logs are archived in
`runs/_logs/`, and each line carries the exact configuration in its `tag=` field:

```bash
grep -h RESULT runs/_logs/*.log        # every sweep number with its config
```

To regenerate one, e.g. the dual-channel comparison:

```bash
python runs/_logs/sweep.py --config configs/scene_graphs_dual.yaml --steps 3000 \
    --tag A train.batch_size=2 train.grad_accum=2
python runs/_logs/sweep.py --config configs/scene_graphs_dual.yaml --steps 3000 \
    --tag B model.num_rounds=0 train.batch_size=2 train.grad_accum=2
```

### Determinism

Runs are seeded (`seed: 0`) and evaluation is deterministic — two evaluations of one checkpoint
give byte-identical predictions (verified). Training is *not* bit-reproducible across different
GPUs or driver versions; expect the last decimal to move.

## Tests

```bash
pytest                                   # tiny-model suite (fast, no GPU model needed)
REGRAPH_LLAMA_TESTS=1 pytest tests/test_llama_real.py   # real-backbone exit criteria
```

Results and per-dataset diagnostics (fusion-gate means, hop distributions, reading entropy) are
reported in `docs/experimental-protocol.md`; every ambiguity resolved during implementation is
logged in `docs/OPEN-QUESTIONS.md`.
