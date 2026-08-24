# StructuralAnomaly

A synthetic, **anchor-free** benchmark for open-ended graph reasoning: locate a region of the
graph by a structural property, then say what that region is *about*.

- **11,000 examples per arm** (8,000 train / 1,000 val / 2,000 test), two arms
- **Chance is 20.00 and analytic**, not estimated
- **Ships with a falsifiable control** that must score chance, and does (19.80)
- No node is named in the question — every node carries `ROLE_NONE`

---

## Why this exists

`ReGraph.md` §2.1 motivates the whole method with an **anchor-free** query:

> For anchor-free questions such as *"Which region of the graph is becoming structurally
> unstable?"* all nodes receive the `none` marker. The model then locates relevant graph regions
> through semantic and structural attention rather than beginning from a predefined question
> entity.

Every benchmark otherwise evaluated in this repo is **anchored**. GraphQA marks the question's
entities as `mentioned`; the text-attributed datasets and NeighborhoodQA mark a centre node as
`source`; NLGraph names both endpoints. The anchor-free path — the capability the specification
leads with — had never been evaluated. This benchmark evaluates it, and nothing else does.

A second reason: every other benchmark here is built from a real corpus, which means its
guessing floor is an empirical accident. NeighborhoodQA's 2-hop variant turned out to have a
48.66 floor because arXiv subject areas co-occur predictably. A synthetic generator lets the
floor be *designed*, proved, and checked.

## The task

A graph holds **k = 5 equally sized thematic communities** (12 nodes each, 60 nodes total). Each
community's nodes carry short text phrases from one theme — *marine biology*, *cryptography*,
*jazz music*, *volcanology*, *textile manufacturing*. Exactly one community is wired far more
densely than the others (`p_hot = 0.60` against `p_in = 0.12`; inter-community `p_out = 0.01`).

> This graph shows a collection of items and the links between them. One group of closely
> related items is far more densely interconnected than any other group. **What is that group
> about?**

The answer is the theme name, generated as free text. No node index, no entity, no anchor.

## Four properties that make it a real test

**1. Chance is provable, and low.** All five themes appear in *every* graph, each is equally
likely to be the dense one, and all communities are the same size. No policy that ignores
topology can exceed 1/k = **20.00%**. This is a fact about the generator, not a measured
baseline — contrast NeighborhoodQA, whose floors had to be discovered after the fact.

**2. Both channels are required.** Topology locates the dense region but cannot name it; the node
text names all five themes but cannot say which one is anomalous. Neither channel alone carries
the answer, which is precisely the `Read` operation the specification describes. Tasks that fail
this test are common: NeighborhoodQA is partly solvable from the centre node's text alone
(72.65 of 83.87), and NLGraph is pure topology with no semantic component at all.

**3. A falsifiable control ships with the data.** `structuralanomaly_control.jsonl.gz` is the
same generator with the density contrast removed (all communities at `p_in`) and the label drawn
uniformly at random. The task is unanswerable by construction, so **any score above chance means
the generator leaks and the main number is void**. ReGraph scores 19.80 on it. This is the check
the other benchmarks in this repo lack.

**4. Difficulty is a dial, not a fixed point.** `density_ratio`, `num_communities`,
`community_size` and the theme vocabularies are all generator parameters. The benchmark yields a
curve; the released setting (`density_ratio = 5.0`) is its easy end.

## Reference results

Llama-3.1-8B-Instruct, frozen, 8 graph-query tokens, 3 reading rounds, ~32.7M trainable
parameters — the default ReGraph configuration, unchanged.

| Configuration | Accuracy | Legality |
| --- | --- | --- |
| Chance (analytic) | 20.00 | — |
| **Control arm** (density contrast removed) | **19.80** | 100.00 |
| **ReGraph, main arm** | **99.05** | 100.00 |

The control landing 0.2 SE from analytic chance is what licenses the main number. Report the
three rows together; 99.05 on its own is not an interpretable result.

**Diagnostics.** The per-round hop weights α of `ReGraph.md` §2.3 collapse onto **hop 2**
(`[0.000, 0.000, 0.999]` in all three rounds) with fusion gates at 0.62 / 0.83 / 0.82 — heavy
evidence injection. On the control arm the gates collapse to 0.05 / 0.10 / 0.11: the model learns
to shut a channel that carries nothing. Across the twelve datasets evaluated in this repo, α is
dataset-specific — hop 0 on NeighborhoodQA, hop 1 on ogbn-arxiv, spread on Cora, hop 2 here.

## Files

| File | |
| --- | --- |
| `structuralanomaly_main.jsonl.gz` | 11,000 examples, the task |
| `structuralanomaly_control.jsonl.gz` | 11,000 examples, the unanswerable control |
| `manifest.json` | task definition, generator parameters, theme vocabulary, reference points |
| `score.py` | standalone scorer, no dependency on this repo |
| `export.py` | regenerates both exports from a preprocessed store |

Each JSONL line:

```json
{"id": 0, "split": "train", "question": "...", "answer": "volcanology",
 "n_nodes": 60, "hot_community": 3,
 "node_text_ids": [12, 7, ...], "edges": [[0, 5], [5, 0], ...]}
```

`node_text_ids` index `manifest.json → node_vocabulary` (61 strings: 5 themes × 12 phrases, plus
the edge text `"linked to"`). Edges are undirected, stored in both directions. `hot_community` is
the index of the dense community in that graph's own community ordering; it is provided for
analysis and is **not** needed to score.

## Scoring

```bash
# predictions.jsonl: one {"id": ..., "pred": "..."} per test example
python score.py predictions.jsonl --arm main
# arm=main  n=2000  accuracy=99.05  legality=100.00
python score.py predictions.jsonl --arm control
# arm=control  n=2000  accuracy=19.80  legality=100.00
```

First-match-wins over the theme vocabulary — the convention this repo uses for Cora/PubMed and
that G-Retriever uses for ExplaGraphs. The scorer warns if the control exceeds chance.

## Limitations

Three, stated plainly.

**The structural signal is degree.** Nodes in the dense community carry about 4.4 more edges than
the rest. Degree is a first-order local quantity that a single round of message passing computes,
so this benchmark demonstrates *anchor-free localization by a local structural cue* — not
path-level reasoning. ReGraph solves it at 99.05 while remaining at chance on NLGraph
connectivity and cycle, and those two facts are consistent: local structure it can read, path
structure it cannot.

**The released setting is saturated.** At `density_ratio = 5.0`, 99.05 is at ceiling and cannot
discriminate between methods. The informative quantity is the accuracy-versus-`density_ratio`
curve and the threshold at which a method falls back to chance. Regenerate at lower ratios
(1.5 / 2.0 / 3.0) to obtain it; the single released point is the easy end.

**Synthetic graphs are not real graphs.** Planted-partition structure, disjoint theme
vocabularies and uniform community sizes are all cleaner than anything real. This benchmark is a
*probe*, meant to isolate one capability under controlled conditions, and should be read
alongside the real-data results rather than instead of them.

## Provenance and regeneration

Generator: [`src/regraph/data/synth_raw.py`](../../src/regraph/data/synth_raw.py). No external
data is downloaded; everything is produced from a seeded `numpy` generator (`seed: 0` in the
configs), so the datasets are reproducible bit-for-bit.

```bash
python -m regraph.data.preprocess_synth --config configs/synth_anomaly.yaml
python -m regraph.data.preprocess_synth --config configs/synth_anomaly_control.yaml
python benchmarks/structuralanomaly/export.py          # rewrites the two .jsonl.gz + manifest

# train and evaluate (each arm ~25 min on one 80 GB GPU)
python -m regraph.train --config configs/synth_anomaly.yaml run_name=seed0
python -m regraph.eval  --config configs/synth_anomaly.yaml \
    --ckpt runs/synth_anomaly/seed0/best.pt --split test

# a difficulty sweep, which is what the benchmark is really for
python -m regraph.data.preprocess_synth --config configs/synth_anomaly.yaml \
    dataset.name=synth_anomaly_r2 dataset.density_ratio=2.0
```
