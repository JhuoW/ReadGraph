# NeighborhoodQA

An **open-ended graph-reasoning** benchmark with exact, programmatically computed ground truth.

Given a paper's sampled citation neighbourhood, name **the set of arXiv CS subject areas
represented among the papers it cites**. The answer is a variable-length list of area names
generated as free text and scored by set-F1 — there is no label set to classify into and no
human or LLM judging is involved.

## Why this exists

It was built to fill a gap found while surveying graph-LLM benchmarks. Methods in this space are
routinely motivated by *non-retrieval, open-ended* graph queries — "which region of the graph is
becoming unstable?" — but the benchmarks in common use do not test that:

| benchmark family | what it actually asks |
|---|---|
| GraphQA (ExplaGraphs / SceneGraphs / WebQSP) | retrieval-style QA; the answer is a span or entity present in the graph |
| Cora / PubMed / arXiv / products | single-label node classification over a fixed vocabulary |
| NLGraph | closed-form structural questions with yes/no or numeric answers |

None requires producing an **aggregate over many nodes** as open-ended text. NeighborhoodQA does.

## The task

```
Q: This is a paper from the arXiv computer science collection, shown with the papers it
   cites. Which arXiv CS subject areas appear among those cited papers? List every area.

A: Data Structures and Algorithms, Robotics, Social and Information Networks
```

Gold answers average **2.28** areas and range from 1 to 8:

| areas in answer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| examples | 7,404 | 8,890 | 5,918 | 2,545 | 857 | 227 | 45 | 8 |

## Three properties that make it a real test

**1. It defeats the single-node shortcut, and the shortcut is measured, not assumed.**
Citation networks are homophilous, so "answer the centre paper's own area" is a strong strategy.
It is quantified: that strategy scores **63.81 set-F1**, which is the floor any result must be
read against. Cora was evaluated as a base first and *rejected* — at 88% homophily the same
shortcut scores 81.5 there, leaving too little headroom to interpret anything.

**2. Ground truth is computed on the subgraph the model actually sees.**
Ego-subgraphs are *sampled* (2 hops, fanout 10). Deriving the gold set from a node's full
neighbourhood would make a large share of examples unanswerable from the shown graph. The label
set is therefore taken from the sampled 1-hop neighbours only.

**3. The output is a set, not a choice.**
Scoring is per-example F1 over parsed area sets, so partial credit is meaningful and the model
must decide *how many* areas to name, not just which one is most likely.

## Reference results

Llama-3.1-8B-Instruct, **frozen**, with ReGraph's graph reader (8 query tokens, 3 reading
rounds). Test split, 3,993 examples.

| configuration | what the model can see | set-F1 | exact-set match |
|---|---|---|---|
| no graph channel (`num_rounds=0`) | nothing | 6.98 | — |
| "answer the centre's own area" | — (analytic) | 63.81 | — |
| **0-hop control** | centre paper's text only | 72.65 | 27.02 |
| **full 2-hop** | centre + neighbours | **83.87** | **45.28** |

**The neighbourhood is worth +11.2 set-F1 (12.3 SE) and +18.3 points of exact-set match.** The
0-hop control is the load-bearing comparison: it is derived from the identical store with each
ego-subgraph truncated to its centre node and the example file reused byte-for-byte, so ground
truth, splits and questions are unchanged and neighbours are the only variable.

Interpret the 6.98 row carefully. The prompt is a *fixed* question, so removing the graph channel
also removes any indication of which paper is being asked about; the model can only emit a
constant guess. That row shows the graph channel carries the signal — it does **not** separate
"read the neighbourhood" from "read the centre node". The 0-hop control is what does that.

## Files

| file | contents |
|---|---|
| `neighborhoodqa.jsonl.gz` | 25,894 examples (19,902 train / 1,999 val / 3,993 test) |
| `manifest.json` | task definition, metric, splits, sampling parameters, reference points |
| `score.py` | standalone scorer, no dependency on this repo |
| `export.py` | regenerates the export from a preprocessed store |

One JSON object per line:

```json
{
  "id": 12345,                  // ogbn-arxiv node index of the centre paper
  "split": "test",
  "question": "...",
  "answer": "Machine Learning, Robotics",   // gold area set, comma-separated
  "n_areas": 2,
  "n_neighbours": 7,
  "center_node": 12345,
  "nodes": [12345, 887, ...],   // local index -> ogbn-arxiv node index; local 0 is the centre
  "edges": [[0, 1], [0, 2], ...]            // local indices
}
```

**Node text is referenced, not inlined**, which keeps the file at 7.4 MB instead of ~1.4 GB and
leaves the choice of text encoder to the user. Recover titles and abstracts from
[`titleabs.tsv`](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz) joined via
OGB's `mapping/nodeidx2paperid.csv.gz`.

## Scoring

```bash
# predictions.jsonl: one {"id": ..., "pred": "..."} per test example
python score.py predictions.jsonl
# n=3993  set_f1=83.87  exact_set_match=45.28
```

Both prediction and gold are parsed into sets by substring-matching the 40 official area names
after normalization (lowercase, punctuation and articles stripped), so ordering, separators and
surrounding prose do not affect the score.

## Limitations

Stated plainly, because they bound what a result on this benchmark means.

* **Homophily makes the centre node informative.** The 0-hop control already reaches 72.65, so
  the task *rewards* graph reading without *depending* on it the way a purely structural task
  would. It measures aggregation over a neighbourhood, not reasoning that is impossible without
  the graph.
* **One source graph.** Everything derives from ogbn-arxiv, so domain diversity is nil and the
  40-area vocabulary is fixed. Output is open-ended in *form* (a variable-length set) but bounded
  in *content*.
* **Sampling is seed-dependent.** Ground truth is tied to the sampled subgraph, so regenerating
  with a different seed produces a different — equally valid — benchmark. Use the released file
  for comparability rather than rebuilding.
* **Supervised setting.** The reference numbers come from training on the train split. They are
  not comparable to zero-shot or prompting results.
* **Small answers.** 63% of examples have 1–2 areas, so set-F1 is coarse-grained on much of the
  data; exact-set-match is reported alongside for that reason.

## Provenance and regeneration

Built from ogbn-arxiv (OGB) with `MIN_NEIGHBOURS = 3`, 2-hop sampling at fanout 10 (mean 56.9
nodes, 71.1 edges, 7.39 one-hop neighbours per example), seed 0. Generator:
`src/regraph/data/nbrqa_raw.py`; config: `configs/arxiv_nbrqa.yaml`.

```bash
python -m regraph.data.preprocess_nbrqa --config configs/arxiv_nbrqa.yaml
python benchmarks/neighborhoodqa/export.py
```

The 0-hop control is derived, not rebuilt, so its ground truth is identical by construction:

```bash
python -m regraph.data.make_zerohop --config configs/arxiv_nbrqa.yaml
```
