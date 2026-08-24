"""NeighborhoodQA — an open-ended graph-reasoning benchmark with computable ground truth.

Why this exists. `ReGraph.md` motivates the method with *non-retrieval, open-ended* queries
("Which region of the graph is becoming structurally unstable?"), but no benchmark evaluated in
this repo actually tests that: GraphQA is retrieval-style QA, the TAG datasets are single-label
classification, and NLGraph is closed-form yes/no. NeighborhoodQA fills the gap.

Task. Given a paper's citation neighbourhood, name **the set of research areas represented
among its neighbours**. The answer is a variable-length list of category names generated as
free text and scored by set-F1 — there is no label to classify into.

Three properties make it a real test rather than a relabelled classification task:

1. **No single-node shortcut.** Answering with the centre paper's own area scores only
   58.8 set-F1 on ogbn-arxiv (measured), so the model must aggregate over neighbours.
2. **Ground truth is computed on exactly the graph the model sees.** Ego-subgraphs are
   *sampled* (fanout 10), so the label set is derived from the sampled 1-hop neighbours, not
   from the full neighbourhood — otherwise the task would be unanswerable by construction.
3. **Open-ended output.** Variable-length set, scored by F1, not accuracy over a fixed vocabulary.

Ground truth is exact (derived from ogbn-arxiv's labels), so no human or LLM judging is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from regraph.data.arxiv_raw import (ARXIV_CS_CATEGORIES, _build_csr, _load_label_names,
                                    _load_node_texts, sample_ego_subgraph)

QUESTION = ("This is a paper from the arXiv computer science collection, shown with the papers "
            "it cites. Which arXiv CS subject areas appear among those cited papers? "
            "List every area that appears.")
QUESTION_HOP2 = ("This is a paper from the arXiv computer science collection, shown with its "
                 "citation neighbourhood. Look at the papers cited by the papers this one "
                 "cites. Which arXiv CS subject areas appear at that second step, excluding "
                 "the ones already cited directly? List every area that appears.")
QUESTION_YEAR = ("This is a paper from the arXiv computer science collection, shown with the "
                 "papers it cites. Each paper's text begins with its publication year in square "
                 "brackets. Considering **only the cited papers published before {year}**, which "
                 "arXiv CS subject areas do those papers cover? List every area that applies.")
EDGE_TEXT = "cites"
MIN_NEIGHBOURS = 3


def build_nbrqa(raw_dir: Path, num_hops: int = 2, fanout: int = 10,
                max_train: int | None = 20000, max_val: int | None = 2000,
                max_test: int | None = 4000, seed: int = 0, target_hop: int = 1,
                year_filter: bool = False):
    """`target_hop=1` -> areas among direct citations (NeighborhoodQA).

    `target_hop=2` -> areas at **exactly** distance 2, excluding those already at distance 1.
    That variant is the first task in this repo that *requires* separating one hop from
    another, i.e. the first real test of the per-token hop weights alpha from `ReGraph.md`
    §2.3, which collapsed onto hop-0 on all ten datasets evaluated so far.

    `year_filter=True` -> NeighborhoodQA-Temporal. Same answer space as `target_hop=1`, but the
    question restricts the aggregation to neighbours published before a per-example cut-off, and
    every node's text is prefixed with its year. The unfiltered answer is a strict superset of
    the gold set, so "ignore the filter and answer everything" has an exactly computable score
    (`attr_blind_setF1` in the returned stats). That makes this the first benchmark here whose
    *failure mode* has a numeric prediction: a model that cannot bind a per-node attribute to its
    node should land at the attribute-blind value, not above it.
    """
    from ogb.nodeproppred import NodePropPredDataset

    ogb_root = Path(raw_dir) / "ogb"
    _tl = torch.load
    torch.load = lambda *a, **k: (k.setdefault("weights_only", False), _tl(*a, **k))[1]
    try:
        ds = NodePropPredDataset(name="ogbn-arxiv", root=str(ogb_root))
    finally:
        torch.load = _tl
    graph, labels = ds[0]
    split = ds.get_idx_split()
    labels = labels.reshape(-1)
    num_nodes = int(graph["num_nodes"])

    label_names = _load_label_names(ogb_root)
    node_texts = _load_node_texts(Path(raw_dir), ogb_root, num_nodes)
    indptr, indices = _build_csr(graph["edge_index"], num_nodes)

    years = None
    if year_filter:
        import gzip as _gz, pandas as _pd
        with _gz.open(ogb_root / "ogbn_arxiv" / "raw" / "node_year.csv.gz", "rt") as f:
            years = _pd.read_csv(f, header=None)[0].to_numpy()
        assert len(years) == num_nodes, (len(years), num_nodes)
        # Surface the attribute in the node text so a failure means the channel cannot BIND it,
        # not that it never saw it -- the same logic as the SceneGraphs coordinate experiment.
        node_texts = [f"[{int(y)}] {t}" for y, t in zip(years, node_texts)]

    rng = np.random.default_rng(seed)
    graphs, examples = {}, []
    shortcut_hits = []
    blind_hits = []
    for name, idx, cap in (("train", split["train"], max_train),
                           ("val", split["valid"], max_val),
                           ("test", split["test"], max_test)):
        idx = np.asarray(idx).reshape(-1)
        eligible = idx[[indptr[v + 1] - indptr[v] >= MIN_NEIGHBOURS for v in idx]]
        if cap is not None and len(eligible) > cap:
            eligible = rng.choice(eligible, size=cap, replace=False)
        for node in eligible:
            node = int(node)
            nodes, ei = sample_ego_subgraph(node, indptr, indices, num_hops, fanout, rng)
            # 1-hop neighbours *within the sampled subgraph* = local ids reached from the centre
            src, dst = ei[0].tolist(), ei[1].tolist()
            hop1 = {int(d) for s, d in zip(src, dst) if s == 0}
            if target_hop == 1:
                targets = sorted(hop1)
            else:
                targets = sorted({int(d) for s, d in zip(src, dst) if s in hop1} - hop1 - {0})
            if len(targets) < MIN_NEIGHBOURS:
                continue
            cutoff = None
            if year_filter:
                ny = [int(years[nodes[j]]) for j in targets]
                cutoff = int(np.median(ny))
                kept = [j for j, y in zip(targets, ny) if y < cutoff]
                full_areas = sorted({label_names[int(labels[nodes[j]])] for j in targets})
                areas = sorted({label_names[int(labels[nodes[j]])] for j in kept})
                # non-empty, and the filter must actually change the answer
                if not areas or set(areas) == set(full_areas):
                    continue
                p = len(set(full_areas) & set(areas)) / len(full_areas)
                r = len(set(full_areas) & set(areas)) / len(areas)
                blind_hits.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
            else:
                areas = sorted({label_names[int(labels[nodes[j]])] for j in targets})
            if target_hop == 2:            # exclude areas already present one hop in
                a1 = {label_names[int(labels[nodes[j]])] for j in hop1}
                areas = sorted(set(areas) - a1)
                if len(areas) < 1:
                    continue
            hop1 = targets
            key = str(node)
            graphs[key] = {
                "node_text_id": torch.tensor(nodes, dtype=torch.int32),
                "edge_text_id": torch.full((ei.shape[1],), num_nodes, dtype=torch.int32),
                "edge_index": ei.to(torch.int32),
            }
            examples.append({
                "id": node, "graph_key": key,
                "question": (QUESTION_YEAR.format(year=cutoff) if year_filter
                             else QUESTION if target_hop == 1 else QUESTION_HOP2),
                "answer": ", ".join(areas), "split": name,
                "roles": [[0, 2]],                        # centre -> ROLE_SOURCE
                "n_areas": len(areas), "n_neighbours": len(hop1),
            })
            own = label_names[int(labels[node])]
            inter = 1.0 if own in areas else 0.0
            p, r = inter / 1.0, inter / len(areas)
            shortcut_hits.append(0.0 if p + r == 0 else 2 * p * r / (p + r))

    stats = {"shortcut_own_topic_setF1": round(float(np.mean(shortcut_hits)) * 100, 2),
             "mean_areas": round(float(np.mean([e["n_areas"] for e in examples])), 2)}
    if blind_hits:      # score of answering the UNFILTERED set, i.e. ignoring the attribute
        stats["attr_blind_setF1"] = round(float(np.mean(blind_hits)) * 100, 2)
    return graphs, examples, node_texts + [EDGE_TEXT], label_names, stats
