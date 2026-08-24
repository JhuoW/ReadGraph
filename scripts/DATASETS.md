# Running ReGraph on each benchmark

Survey of the benchmarks used by the related work, which of them fit ReGraph, and the exact
commands for the ones this repo supports. **Always `source scripts/env.sh` first** — it points
`HF_HOME` at `/mnt/ssd1/zhuowei/hf-cache`, where the Llama backbone lives.

## Which benchmarks fit ReGraph, and why

ReGraph needs four things: a **per-example graph** (or an ego-subgraph of a big one), a
**natural-language instruction**, an **open-ended text answer**, and a **white-box frozen LLM**
whose residual stream can be edited mid-stack. Two further facts from our own measurements
(`docs/experimental-protocol.md`) decide fit in practice:

* the reader softmaxes over every node, and its localization was measured to collapse past
  ~1,371 nodes — so big graphs need ego-subgraph sampling;
* the graph→LLM channel carries **coarse semantics** (a class, a yes/no) but not **identity or
  surface form** (an entity name, one object's attribute).

| Benchmark | Source paper | Fit | Status |
|---|---|---|---|
| GraphQA: ExplaGraphs / SceneGraphs / WebQSP | G-Retriever (NeurIPS'24) | good | **done** |
| ogbn-arxiv | GraphTranslator (WWW'24), also LLaGA, GraphGPT | good | **done** |
| NLGraph: connectivity, cycle | Can LMs Solve Graph Problems? (NeurIPS'23) | **best fit for the paper's own claim** | **done** |
| NLGraph: shortest path, topo sort, Hamilton, matching, flow, GNN | same | poor — answers are node sequences/numbers, i.e. surface form | not run (see below) |
| Talk like a Graph | Fatemi et al. (ICLR'24) | overlaps NLGraph; same task family, synthetic graphs | not run |
| Cora / PubMed | GraphGPT (SIGIR'24), LLaGA (ICML'24) | good — same shape as arxiv | **run** — Cora 86.72 / 88.99 (3B), PubMed 89.98; see README Tables 4-5 |
| **StructuralAnomaly** (this repo) | none — synthetic | **anchor-free**, the one path `ReGraph.md` §2.1 leads with | **run** — 99.05 vs analytic chance 20.00, control 19.80; see `benchmarks/structuralanomaly/` |
| ogbn-products | LLaGA | good, but 2.4M nodes → heavy preprocessing | not run |
| Taobao | GraphTranslator | **impossible** — data not public, human/ChatGPT-rated | blocked |
| GRAFF's four datasets | GRAFF (Findings EACL'26) | 3 of 4 are GraphQA (already run); comparison blocked by retrieval + backbone confounds | **see below** |
| WikiWeb2M | Multimodal Graph Learning | out of scope — multimodal (images + text) | not run |

The **NLGraph** pair is the most important addition. `ReGraph.md` §2.1 draws its own worked
example from this task family ("Find a path from node 7 to node 19"), and the paper opens by
arguing GraphRAG "can not handle non-retrieval query" — a claim no retrieval-style benchmark
tests. Connectivity supplies an unambiguous `source`/`target` pair (the first use of
`ROLE_TARGET` in the project) and cycle is the anchor-free case §2.1 calls "a first-class case".

---

## 1. GraphQA — ExplaGraphs, SceneGraphs, WebQSP

Raw data: copy G-Retriever's `train_dev.tsv` to `data/raw/expla_graphs/`, and `questions.csv` +
`sceneGraphs.zip` to `data/raw/scene_graphs/`. WebQSP downloads from HF (`rmanluo/RoG-webqsp`).

```bash
source scripts/env.sh
for DS in expla_graphs scene_graphs webqsp; do
  python -m regraph.data.preprocess --config configs/$DS.yaml
  python -m regraph.train          --config configs/$DS.yaml run_name=seed0
  python -m regraph.eval           --config configs/$DS.yaml \
      --ckpt runs/$DS/seed0/best.pt --split test --dump-readings 5
done
```

Preprocessing self-checks the split sizes and average node/edge counts against
`docs/experimental-protocol.md` and refuses to continue on a mismatch.

## 2. ogbn-arxiv (GraphTranslator / LLaGA / GraphGPT)

Needs `ogbn-arxiv` via OGB plus `titleabs.tsv` for raw text, and GraphTranslator's repo for its
4,000-node evaluation subset:

```bash
source scripts/env.sh
mkdir -p /mnt/ssd1/zhuowei/regraph-cache/arxiv_raw && cd $_
hf download Hualouz/GraphTranslator-arixv --repo-type dataset --include "titleabs.tsv" --local-dir .
python -c "from ogb.nodeproppred import NodePropPredDataset; NodePropPredDataset('ogbn-arxiv', root='ogb')"
git clone https://github.com/alibaba/GraphTranslator.git   # for arxiv_test_idx_random_4000.csv
cd -

python -m regraph.data.preprocess_arxiv --config configs/arxiv.yaml
python -m regraph.train                 --config configs/arxiv.yaml run_name=control
python -m regraph.eval                  --config configs/arxiv.yaml \
    --ckpt runs/arxiv/control/best.pt --split test --dump-readings 5
python -m regraph.eval_rank             --config configs/arxiv.yaml \
    --ckpt runs/arxiv/control/best.pt          # Top-3 / Top-5, to match their table
```

Optional two-stage variant (alignment pretraining — **tested, no effect**, see README):

```bash
python -m regraph.data.add_align_targets --config configs/arxiv.yaml
python -m regraph.train --config configs/arxiv_align.yaml
python -m regraph.train --config configs/arxiv.yaml run_name=aligned \
    --init-from runs/arxiv/align/best.pt
```

Point `dataset.graphtranslator_test_subset` at your clone of that CSV, and
`data.arxiv_raw_dir` at wherever you put the raw files.

## 3. NLGraph — connectivity and cycle

```bash
source scripts/env.sh
git clone https://github.com/Arthur-Heng/NLGraph.git      # set data.nlgraph_raw_dir to <clone>/NLGraph
for T in connectivity cycle; do
  python -m regraph.data.preprocess_nlgraph --config configs/nlgraph_$T.yaml
  python -m regraph.train                   --config configs/nlgraph_$T.yaml run_name=seed0
  python -m regraph.eval                    --config configs/nlgraph_$T.yaml \
      --ckpt runs/nlgraph_$T/seed0/best.pt --split test
done
python scripts/nlgraph_report.py            # per-difficulty breakdown vs their Table 2
```

Notes that affect how the numbers should be read:

* NLGraph puts the edge list **in the prompt**; ReGraph strips it and feeds topology through the
  encoder. That is the point of the method, but it means the input is not identical to theirs.
* Nodes are bare integers, so every node shares one placeholder embedding and is distinguishable
  only by topology and role marker.
* **Their baselines are zero-/few-shot prompting; ReGraph is supervised** on the train split.
  Not a like-for-like comparison — state it with every number.
* `reader.max_hops` (K=2 by default) bounds how far diffusion reaches. Connectivity on the hard
  subset may need a larger K; it is a config value, not a code change:
  `python -m regraph.train --config configs/nlgraph_connectivity.yaml reader.max_hops=4`

## 4. Not implemented, and what each would take

**Cora / PubMed** (GraphGPT, LLaGA) — **implemented and run**; see `tag_raw.py` and
`configs/{cora,pubmed}.yaml`. The loader uses the TAPE release for raw titles/abstracts, because
PyG's Planetoid ships bag-of-words only, which loses the text ReGraph's attribute encoder needs.
Note the Cora node-ID trap documented in README Table 4.

**ogbn-products** (LLaGA) — 2.4M nodes; the ego-subgraph sampler in `arxiv_raw.py` already
handles this shape, but expect a long attribute-encoding pass.

**NLGraph's remaining six tasks** — shortest path, topological sort, Hamilton path, bipartite
matching, max flow and GNN simulation all require emitting node sequences or exact numbers.
That is precisely the surface-form regime the channel was measured to fail at (WebQSP entities
62.22, SceneGraphs colour 28.1), so a poor score there would confirm a known limitation rather
than reveal a new one. Worth running only as a deliberate negative result.

**Talk like a Graph** (Fatemi et al.) — same synthetic-structural family as NLGraph, and its
contribution is about *text encodings of graphs*, which ReGraph bypasses entirely. Low marginal
information over NLGraph.

**Taobao** (GraphTranslator) — not obtainable. The authors released only
`Hualouz/GraphTranslator-arxiv`; Taobao is 980k Alibaba users with behavioural data and its
open-ended tasks are scored by human/ChatGPT rating, so neither the data nor the metric is
reproducible.

**WikiWeb2M** (Multimodal Graph Learning) — nodes carry images as well as text. ReGraph's
attribute encoder is text-only; §2.1 does allow visual attributes in principle, so this is a
genuine extension rather than a mismatch, but it needs a vision encoder.


---

## 5. GRAFF (Findings of EACL 2026) — already covered, but not comparable

GRAFF evaluates on ExplaGraphs, SceneGraphs and WebQSP — all three already in this repo — plus a
`SyntheticGraph` variant that is **not released** (their repo is an empty stub; the paper
describes it only as shuffling ExplaGraphs nodes and arguments while preserving edges).

Two confounds block a direct comparison, both documented in README.md:

* GRAFF evaluates on **PCST-retrieved subgraphs** (their Table 2: WebQSP 8.39 avg nodes) while
  ReGraph reads the **full graph** (~1,371 nodes) per `ReGraph.md` §3.2.
* Their backbone is LLaMA-3.2-3B; ours is Llama-3.1-8B-Instruct.

A retrieval stage exists for closing the first gap:

```bash
python -m regraph.data.pcst_retrieve --config configs/webqsp.yaml --topk 3 --topk-e 5
python -m regraph.train --config configs/webqsp_pcst.yaml run_name=seed0
python -m regraph.eval  --config configs/webqsp_pcst.yaml --ckpt runs/webqsp_pcst/seed0/best.pt --split test
```

**It is not usable as-is, and the reason is recorded rather than hidden.** `pcst_retrieve` is a
verified-faithful port of G-Retriever's `retrieval_via_pcst` (identical node/edge counts to the
original on sampled graphs), but with G-Retriever's own WebQSP settings it returns 2.54 nodes
per graph and the gold answer survives in only 13-15% of examples, versus 95% in the full graph.
Sweeping `--topk` does not fix it. Anyone re-attempting this should first re-run the recall
check before training:

```python
# answer entity present in retrieved subgraph, vs in the full graph
# see README "Comparison with GRAFF" for the measured numbers
```

---

## 6. Dual-channel mode (token channel + reader) — the augmentation that worked

Serializes the graph into the prompt *in addition to* running the Read-Fuse-Replace rounds.
Architecture is untouched; only `data.serialize_graph` changes behaviour.

```bash
source scripts/env.sh
# both channels
python -m regraph.train --config configs/scene_graphs_dual.yaml run_name=dual \
    train.batch_size=2 train.grad_accum=2          # sequences are ~1,500 tokens
python -m regraph.eval  --config configs/scene_graphs_dual.yaml \
    --ckpt runs/scene_graphs/dual/best.pt --split test

# the control that makes the result interpretable: token channel WITHOUT the reader
python -m regraph.train --config configs/scene_graphs_dual.yaml run_name=textonly \
    model.num_rounds=0 train.batch_size=2 train.grad_accum=2
```

Measured at 3,000 steps: 34.93 (reader only) → 74.93 (token only) → **81.93 (both)**;
the reader contributes **+7.00 pp** over token-only at 4.6 SE. Always run the `num_rounds=0`
control — without it you cannot tell which channel produced the gain.

Applying this to another dataset needs only `data.serialize_graph: true` plus a
`max_question_tokens` large enough for the rendered graph. Check two things first: that PCST
retrieval (if you use it) preserves the answer, and that the question survives truncation —
decode a real training prompt rather than trusting the flag.
